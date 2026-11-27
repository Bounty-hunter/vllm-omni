# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selection and fallback on supported diffusion layouts."""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.parallel.head_overlap.head_candidates import candidates, eligible
from vllm_omni.diffusion.attention.parallel.head_overlap.head_policy import candidate_index, choose

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dense_video_eligibility():
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend

    owner = SimpleNamespace(causal=False, effective_attention=lambda: (None, FlashAttentionBackend, None))
    kernel = MethodType(lambda self: None, owner)
    grid = (9, 30, 52)
    rows = 14040
    layout = VideoTokenLayout(prefix_len=0, latent_grid=grid, used_len=rows)
    md = AttentionMetadata(video_layout=layout, extra={"vsa_dit_seq_shape": grid})
    assert eligible(kernel, md, rows)
    assert eligible(kernel, AttentionMetadata(), rows)
    assert not eligible(kernel, md, rows + 4)
    for unsupported in (
        replace(md, attn_mask=torch.ones(1, rows, dtype=torch.bool)),
        replace(md, extra={**md.extra, "kv_cache_dtype": "fp8"}),
        replace(md, extra={"vsa_dit_seq_shape": (9, 30, 51)}),
        replace(md, video_layout=replace(layout, prefix_len=1)),
        replace(md, video_layout=replace(layout, used_len=rows - 1)),
    ):
        assert not eligible(kernel, unsupported, rows)
    owner.causal = True
    assert not eligible(kernel, md, rows)
    owner.causal = False
    owner.effective_attention = lambda: (None, None, None)
    assert not eligible(kernel, md, rows)


def test_candidate_selection_and_fallback():
    assert choose([[100] * 10, [97] * 10]) == 0
    assert choose([[100] * 10, [90] * 10]) == 1
    assert choose([[100] * 5, [80] * 5]) == 0
    assert [candidate_index(i) for i in range(8)] == [0, 1, 1, 0, 1, 0, 0, 1]
    for heads in (1, 5, 8, 10, 16):
        for rows in (512, 1024, 4096, 14040):
            choices = candidates(heads, rows, 120)
            assert len(choices) <= 6 and all(sum(parts) == heads and min(parts) > 0 for parts in choices)
            assert [len(parts) for parts in choices] == sorted((len(parts) for parts in choices), reverse=True)
            for parts in choices:
                if len(parts) == 3:
                    assert parts[0] == parts[-1] <= parts[1]
    assert candidates(8, 512, 120) == ()
    assert candidates(8, 15360, 120) == ((1, 6, 1), (2, 4, 2), (1, 7))
    assert candidates(2, 15360, 120) == ((1, 1),)
    # Qwen 1536^2: the three-group estimate adds one wave. It must reach
    # real-layer sampling rather than silently becoming a two-group-only test.
    assert candidates(16, 9216, 132) == ((1, 14, 1), (2, 12, 2), (3, 10, 3), (4, 8, 4), (5, 6, 5), (5, 11))


def test_mismatched_layer_order_skips_collective_preparation(monkeypatch):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(head_policy.dist, "get_world_size", lambda group: 2)

    def gather(peers, local, group):
        peers[:] = [local, list(reversed(local))]

    monkeypatch.setattr(head_policy.dist, "all_gather_object", gather)
    policy = head_policy.HeadPolicy(None, 120)
    policy.component_key = "shape"
    policy.discovered["layout"] = head_policy.Trial((3, 5), example=("cuda", "pg"))
    policy.discovery = [("first", "layout"), ("second", "layout")]
    policy.finish_component(None, (), None)
    assert policy.visits["shape"] == {}
    assert not policy.layouts and not policy.pending and not policy.discovered


@pytest.mark.parametrize("failure,recoverable", [(MemoryError("allocation"), True), (ValueError("bad layout"), False)])
def test_prepare_local_votes_before_rendezvous(monkeypatch, failure, recoverable):
    from vllm_omni.diffusion.attention.parallel import ulysses_ce_overlap as ce

    def prepare():
        raise failure

    def gather(peers, local, group):
        peers[:] = [None, local]

    monkeypatch.setattr(ce.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(ce.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match=str(failure)) as caught:
        ce.prepare_local(prepare, object())
    assert isinstance(caught.value, ce.PreparationError) is recoverable


@pytest.mark.parametrize("peer_split_us,expected", [(65.0, True), (105.0, False)])
def test_one_decision_from_slowest_rank(monkeypatch, peer_split_us, expected):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy as hp

    policy = hp.HeadPolicy(None, 120)
    key = (2, (((1, 127, 16, 64),),))
    trial = hp.Trial((3, 5), plan=SimpleNamespace(widths=(3, 5)), seen=hp._UNTIMED + hp._SAMPLES)
    policy.component_layouts["request"] = (key,)
    policy.layouts[key] = trial
    policy.pending.add(key)
    end = SimpleNamespace(query=lambda: True)
    for i in range(hp._SAMPLES):
        ms = 0.06 if hp.candidate_index(i) else 0.10
        trial.observations[i] = SimpleNamespace(elapsed_time=lambda _, ms=ms: ms), end, True

    calls = []

    def gather(peers, local, group):
        calls.append(1)
        peer = [(peer_split_us if hp.candidate_index(i) else 100.0, True) for i in range(hp._SAMPLES)]
        peers[:] = [local, {key: peer}]

    monkeypatch.setattr(hp.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(hp.dist, "all_gather_object", gather)
    policy.resolve_pending("request")
    assert trial.selected is expected and not policy.pending and not trial.observations
    policy.resolve_pending("request")
    assert len(calls) == 1


@pytest.mark.parametrize("triple_us,pair_us,expected", [(80, 70, (1, 6, 1)), (103, 85, (1, 7)), (103, 102, None)])
def test_three_groups_before_two_without_new_resources(monkeypatch, triple_us, pair_us, expected):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy as hp

    key = (2, (((1, 127, 16, 64),),))
    policy = hp.HeadPolicy(None, 120)
    triple = SimpleNamespace(widths=(1, 6, 1))
    pair = SimpleNamespace(widths=(1, 7))
    timers = [object() for _ in range(hp._SAMPLES)]
    trial = hp.Trial(triple.widths, plan=triple, fallback_plans=[pair], timers=timers)
    policy.layouts[key] = trial
    policy.component_layouts["request"] = (key,)
    policy.pending.add(key)
    monkeypatch.setattr(hp.dist, "get_world_size", lambda group: 2)
    calls = []

    def gather(peers, local, group):
        calls.append(1)
        peers[:] = [local, local]

    def unexpected(*args, **kwargs):
        pytest.fail("Phase transition must not prepare resources or synchronize")

    monkeypatch.setattr(hp.dist, "all_gather_object", gather)
    monkeypatch.setattr(policy, "_prepare", unexpected)
    monkeypatch.setattr(torch.accelerator, "synchronize", unexpected)
    for cost in (triple_us, pair_us):
        trial.seen = hp._UNTIMED + hp._SAMPLES
        for i in range(hp._SAMPLES):
            us = cost if hp.candidate_index(i) else 100.0
            trial.observations[i] = (
                SimpleNamespace(elapsed_time=lambda _, us=us: us / 1000),
                SimpleNamespace(query=lambda: True),
                True,
            )
        policy.resolve_pending("request")
        if trial.selected is not None:
            break
        assert trial.plan is pair and trial.widths == (1, 7) and trial.seen == 0
        assert trial.timers is timers and len(timers) == hp._SAMPLES
        assert not trial.observations and key in policy.pending
    assert (trial.widths if trial.selected else None) == expected
    assert not policy.pending and not trial.timers and not trial.fallback_plans
    assert len(calls) == (1 if expected == (1, 6, 1) else 2)
    policy.resolve_pending("request")
    assert len(calls) == (1 if expected == (1, 6, 1) else 2)


@pytest.mark.parametrize(
    "costs,expected,phases",
    [
        ([(90, 90), (80, 80), (70, 70), (65, 65), (60, 60), (50, 50)], (5, 6, 5), 5),
        ([(80, 80), (95, 95), (70, 70), (90, 90), (85, 85), (50, 50)], (3, 10, 3), 5),
        ([(80, 80), (79, 79), (70, 70), (69, 69), (60, 60), (50, 50)], (5, 6, 5), 5),
        ([(110, 110), (109, 109), (80, 80), (90, 90), (95, 95), (50, 50)], (3, 10, 3), 5),
        ([(110, 110), (109, 109), (108, 108), (107, 107), (106, 106), (80, 80)], (5, 11), 6),
        ([(110, 110), (109, 109), (108, 108), (107, 107), (106, 106), (102, 102)], None, 6),
        ([(80, 80), (95, 95), (90, 90), (50, 50), (40, 40), (30, 30)], (1, 14, 1), 3),
        ([(80, 80), (70, 70), (90, 90), (50, 50), (40, 40), (30, 30)], (2, 12, 2), 3),
        # The faster rank cannot keep promoting a candidate that regresses on its peer.
        ([(80, 90), (60, 120), (60, 110), (50, 50), (40, 40), (30, 30)], (1, 14, 1), 3),
    ],
)
def test_search_compares_incumbent_and_stops_after_regression(monkeypatch, costs, expected, phases):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy as hp

    key = (2, (((1, 4608, 32, 128),),))
    widths = ((1, 14, 1), (2, 12, 2), (3, 10, 3), (4, 8, 4), (5, 6, 5), (5, 11))
    monkeypatch.setattr(hp.dist, "get_world_size", lambda group: 2)

    def unexpected(*args, **kwargs):
        pytest.fail("Search phases must not allocate resources or synchronize the device")

    monkeypatch.setattr(hp.HeadPolicy, "_prepare", unexpected)
    monkeypatch.setattr(torch.accelerator, "synchronize", unexpected)
    outcomes = []
    for rank in range(2):
        policy = hp.HeadPolicy(None, 132)
        plans = [SimpleNamespace(widths=w) for w in widths]
        timers = [object() for _ in range(hp._SAMPLES)]
        trial = hp.Trial(widths[0], plan=plans[0], fallback_plans=plans[1:], timers=timers)
        policy.layouts[key] = trial
        policy.component_layouts["request"] = (key,)
        policy.pending.add(key)
        calls = []
        while policy.pending:
            index = widths.index(trial.widths)
            reference = costs[widths.index(trial.best_plan.widths)] if trial.best_plan else (100, 100)
            reports = [
                {
                    key: [
                        (costs[index][peer] if hp.candidate_index(i) else reference[peer], True)
                        for i in range(hp._SAMPLES)
                    ]
                }
                for peer in range(2)
            ]

            def gather(peers, local, group):
                calls.append(1)
                assert local == reports[rank]
                peers[:] = reports

            monkeypatch.setattr(hp.dist, "all_gather_object", gather)
            trial.seen = hp._UNTIMED + hp._SAMPLES
            for i, (us, queued) in enumerate(reports[rank][key]):
                trial.observations[i] = (
                    SimpleNamespace(elapsed_time=lambda _, us=us: us / 1000),
                    SimpleNamespace(query=lambda: True),
                    queued,
                )
            policy.resolve_pending("request")
            assert not trial.observations
            if policy.pending:
                assert trial.timers is timers and len(timers) == hp._SAMPLES and trial.seen == 0
            assert len(calls) <= len(widths)
        outcomes.append(trial.widths if trial.selected else None)
        assert outcomes[-1] == expected and len(calls) == phases
        assert not trial.timers and not trial.fallback_plans and trial.best_plan is None
        policy.resolve_pending("request")
        assert len(calls) == phases
    assert outcomes[0] == outcomes[1]
