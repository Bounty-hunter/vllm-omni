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
    for count in range(2, 10):
        for cycle in range(6):
            assert sorted(candidate_index(cycle * count + i, count) for i in range(count)) == list(range(count))
    for heads in range(2, 17):
        choices = candidates(heads, 9216, 132)
        assert len(choices) == heads // 2
        assert all(len(p) == 2 and 0 < p[0] <= p[1] and sum(p) == heads for p in choices)
    assert candidates(8, 512, 132) == ()
    assert candidates(32, 9216, 132) == ()
    assert candidates(16, 9216, 132)[:2] == ((5, 11), (7, 9))


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


def make_trial(hp, *, ready=True):
    key = (2, (((1, 4608, 32, 128),),))
    plans = tuple(SimpleNamespace(widths=w) for w in ((5, 11), (7, 9)))
    count = hp._ROTATIONS * 3
    trial = hp.Trial((5, 11), plan=plans[0], plans=plans, timers=[None] * count, seen=count, commit_at=6)
    state = {"ready": ready, "syncs": 0}

    def synchronize():
        state["syncs"] += 1
        state["ready"] = True

    end = SimpleNamespace(query=lambda: state["ready"], synchronize=synchronize)
    for i in range(count):
        cost = (100, 70, 80)[hp.candidate_index(i, 3)]
        trial.observations[i] = (SimpleNamespace(elapsed_time=lambda _, c=cost: c / 1000), end, True)
    return key, trial, state


@pytest.mark.parametrize("peer_cost,expected", [(75, (5, 11)), (110, (7, 9))])
def test_async_max_commits_at_same_boundary(monkeypatch, peer_cost, expected):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy as hp

    policy = hp.HeadPolicy(None, 132)
    key, trial, state = make_trial(hp, ready=False)
    policy.layouts[key] = trial
    policy.pending.add(key)
    calls = []
    waits = []

    def reduce(tensor, op, group, async_op):
        assert async_op and op == hp.dist.ReduceOp.MAX
        calls.append(tensor.clone())
        peer = torch.tensor([(100, peer_cost, 85)[hp.candidate_index(i, 3)] for i in range(len(tensor))])
        tensor.copy_(torch.maximum(tensor, peer))
        return SimpleNamespace(wait=lambda: waits.append(1))

    monkeypatch.setattr(hp.dist, "all_reduce", reduce)
    monkeypatch.setattr(hp.dist, "all_gather_object", lambda *a, **kw: pytest.fail("No per-step object gather"))
    policy.step = 4
    policy.resolve_pending()
    assert not calls and state["syncs"] == 0
    state["ready"] = True
    policy.resolve_pending()
    assert len(calls) == 1 and not waits and trial.selected is None
    policy.step = 5
    policy.resolve_pending(commit=True)
    assert len(calls) == 1 and not waits and trial.selected is None
    policy.step = 6
    policy.resolve_pending(commit=True)
    assert trial.selected and trial.widths == expected
    assert len(waits) == 1 and not policy.pending and not trial.timers and not trial.observations
    policy.resolve_pending(commit=True)
    assert len(calls) == len(waits) == 1


def test_late_rank_waits_only_at_fixed_commit(monkeypatch):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy as hp

    policy = hp.HeadPolicy(None, 132)
    key, trial, state = make_trial(hp, ready=False)
    policy.layouts[key] = trial
    policy.pending.add(key)
    calls = []
    monkeypatch.setattr(hp.dist, "all_reduce", lambda *a, **kw: calls.append(1) or SimpleNamespace(wait=lambda: None))
    for step in (4, 5):
        policy.step = step
        policy.resolve_pending(commit=True)
        assert not calls and state["syncs"] == 0
    policy.step = 6
    policy.resolve_pending(commit=True)
    assert state["syncs"] == 1 and calls == [1] and trial.selected


def test_request_reset_forgets_choice_but_keeps_resources(monkeypatch):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy as hp

    policy = hp.HeadPolicy(None, 132)
    key, trial, _ = make_trial(hp)
    policy.layouts[key] = trial
    policy.pending.add(key)
    resources = policy.prepared = object()
    workspace = policy.workspace = object()
    waits = []
    monkeypatch.setattr(hp.dist, "all_reduce", lambda *a, **kw: SimpleNamespace(wait=lambda: waits.append(1)))
    policy.end_request()
    assert waits == [1] and trial.selected is None
    policy.begin_request()
    assert not policy.layouts and not policy.pending and policy.step == 0
    assert policy.prepared is resources and policy.workspace is workspace


def test_missing_visits_fall_back(monkeypatch):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy as hp

    policy = hp.HeadPolicy(None, 132)
    key, trial, _ = make_trial(hp)
    for i in range(0, len(trial.timers), 3):
        trial.observations.pop(i)
    policy.layouts[key] = trial
    policy.pending.add(key)
    monkeypatch.setattr(hp.dist, "all_reduce", lambda *a, **kw: SimpleNamespace(wait=lambda: None))
    policy.step = 6
    policy.resolve_pending(commit=True)
    assert trial.selected is False and trial.plan is None and trial.widths == (16,)
