# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for conservative selection and balanced natural-call sampling."""

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

from lazy_head_policy import (
    candidate_index,
    candidates,
    choose,
    choose_three_then_two,
    metadata_spec,
    qualified_cycles,
    symmetric_candidates,
    tensor_spec,
    three_then_two_candidates,
    two_pass_order,
)


class TensorLayout:
    def __init__(self, stride=1):
        self.shape, self.dtype, self.device, self.pitch = (3,), "int32", SimpleNamespace(type="cuda"), stride

    def stride(self):
        return (self.pitch,)

    def item(self):
        raise AssertionError("Device data must never be read by signature construction")


@dataclass
class Packed:
    q_length: int
    boundaries: object


@dataclass
class Metadata:
    packing: Packed
    extra: dict


def main():
    def attention_layout(batch_stride, row_stride):
        return SimpleNamespace(
            shape=(1, 17, 32, 128),
            dtype="bf16",
            device=SimpleNamespace(type="cuda"),
            stride=lambda: (batch_stride, row_stride, 128, 1),
        )

    assert tensor_spec(attention_layout(1100 * 4096, 4096)) == tensor_spec(attention_layout(17 * 4096, 4096))
    assert tensor_spec(attention_layout(17 * 4096, 4096)) != tensor_spec(attention_layout(17 * 12288, 12288))
    first = Metadata(Packed(9248, TensorLayout()), {"max_seqlen": 9248})
    same = Metadata(Packed(9248, TensorLayout()), {"max_seqlen": 9248})
    assert metadata_spec(first) == metadata_spec(same) and metadata_spec(first)[1]
    assert metadata_spec(first) != metadata_spec(Metadata(Packed(4128, TensorLayout()), first.extra))
    assert metadata_spec(first) != metadata_spec(Metadata(Packed(9248, TensorLayout(2)), first.extra))
    assert not metadata_spec(object())[1]
    assert not metadata_spec({"nested": [object()]})[1]
    first = [[(99, True, 1), (101, True, 1)], [(100, True, 1), (103, True, 1)]]
    second = [[(80, True, 1), (82, True, 1)], [(81, True, 1), (83, True, 1)]]
    assert qualified_cycles([first, second]) == [[103], [83]]
    assert qualified_cycles([first, [second[0], [(81, True, 1), (83, False, 1)]]]) == [[], []]
    assert qualified_cycles([first + [[(99, True, 1)]], second]) == [[103], [83]]
    for heads in (10, 12, 14):
        for tokens in (288, 1056, 1344, 2048, 7800, 8128, 9248, 14040, 16416, 20623, 26806, 47286):
            choices = candidates(heads, tokens, 128, 4 if heads != 12 else 2, 132)
            assert choices[0] == (heads,) and len(choices) <= 3
            assert len(set(choices)) == len(choices)
            assert all(sum(widths) == heads and min(widths) > 0 for widths in choices)
    assert any(any(width % 2 for width in widths) for widths in candidates(14, 9248, 128, 4, 132)[1:])
    assert symmetric_candidates(8) == ((8,), (1, 6, 1), (2, 4, 2))
    assert symmetric_candidates(14) == ((14,), (1, 12, 1), (2, 10, 2))
    for heads in range(1, 33):
        choices = symmetric_candidates(heads)
        assert choices[0] == (heads,) and len(choices) <= 3
        assert all(sum(parts) == heads and min(parts) > 0 for parts in choices)
        assert all(parts[0] == parts[-1] <= parts[1] for parts in choices[1:])
    assert Counter(candidate_index(i, 3) for i in range(24)) == {0: 8, 1: 8, 2: 8}
    for count in (1, 2, 3):
        cycle = [candidate_index(i, count) for i in range(2 * count * count)]
        assert Counter(cycle) == {i: 2 * count for i in range(count)}
        for position in range(count):
            assert Counter(cycle[position::count]) == {i: 2 for i in range(count)}
    assert Counter(candidate_index(i, 4) for i in range(16)) == {2: 8, 3: 8}
    assert Counter(candidate_index(i, 4) for i in range(16, 32)) == {0: 8, 1: 8}
    for offset in (0, 16):
        for position in (0, 1):
            assert len(set(candidate_index(i, 4) for i in range(offset + position, offset + 16, 2))) == 2
    assert choose([[100] * 10, [99] * 10]) == 0
    assert choose([[100] * 5, [70] * 5]) == 0
    assert choose([[100] * 10, [80] * 10, [90] * 10]) == 1
    assert choose([[100] * 10, [110] * 10]) == 0
    assert choose([[100] * 10, [50, 150] * 5]) == 0
    preferred = ((8,), (1, 6, 1), (3, 5))
    assert choose_three_then_two([[1000] * 7, [800] * 7, [795] * 7], preferred) == 1
    assert choose_three_then_two([[1000] * 7, [900] * 7, [800] * 7], preferred) == 2
    assert choose_three_then_two([[1000] * 7, [1100] * 7, [900] * 7], preferred) == 2
    assert choose_three_then_two([[1000] * 5, [700] * 5, [800] * 5], preferred) == 0
    assert choose_three_then_two([[1000] * 7, [1100] * 7, [1200] * 7], preferred) == 0
    assert choose_three_then_two([[1000] * 7, [800] * 7], ((8,), (3, 5))) == 1
    paired = {
        "whole_vs_three": [[], []],
        "whole_vs_two": [[1000] * 7, [800] * 7],
        "three_vs_two": [[], []],
    }
    assert choose_three_then_two([[], [], []], preferred, paired) == 2
    controlled = (preferred[0], preferred[1], preferred[0], preferred[2])
    assert choose_three_then_two([[1000] * 7, [800] * 7, [1000] * 7, [795] * 7], controlled) == 1
    assert choose_three_then_two([[1000] * 7, [1100] * 7, [1000] * 7, [800] * 7], controlled) == 3
    assert choose_three_then_two([[1000] * 7, [800] * 7, [1200] * 7, [900] * 7], controlled) == 3
    assert choose_three_then_two([[], [], [], []], controlled, paired) == 3
    for calls in (32, 40, 42):
        previous = Counter(preferred[candidate_index(i, 3)] if i < 24 else preferred[0] for i in range(calls))
        current = Counter(controlled[candidate_index(i, 4)] if i < 32 else controlled[0] for i in range(calls))
        assert current == previous  # Reorder the same real cores, never add work.
        buffered = Counter(
            controlled[candidate_index(i - 2, 4)] if 2 <= i < 32 else controlled[0] for i in range(calls)
        )
        assert sum(buffered.values()) == calls
        assert buffered[preferred[2]] == 8 and buffered[preferred[1]] == 7
        assert sum(len(parts) * count for parts, count in buffered.items()) < sum(
            len(parts) * count for parts, count in previous.items()
        )
    # R5 lost three initial double pairs to launch starvation and incorrectly
    # preferred a weaker triple. Leading real whole layers can admit one more
    # complete pair without lowering the six-pair confidence requirement.
    paired = {
        "whole_vs_three": [[1110] * 6, [1050] * 6],
        "whole_vs_two": [[1090] * 6, [960] * 6],
        "three_vs_two": [[1050] * 6, [960] * 6],
    }
    samples = [[1110] * 6, [1050] * 6, [1090] * 6, [960] * 6]
    assert choose_three_then_two(samples, controlled, paired) == 3
    paired["whole_vs_two"] = [[1090] * 5, [960] * 5]
    assert choose_three_then_two(samples, controlled, paired) == 1
    order = two_pass_order(32, 64)
    assert len(order) == 64 and order[:2] == (0, 0)
    assert set(order[2:32]) == {0, 1} and set(order[32:]) == {2, 3}
    assert Counter(order) == {0: 17, 1: 15, 2: 16, 3: 16}
    # Six early host-starved layers still leave enough complete, all-rank
    # observations. Rejecting them preserves the same confidence gate.
    observations = [[] for _ in controlled]
    for index, choice in enumerate(order[2:], 2):
        value = (1100, 1050, 1090, 960)[choice]
        observations[choice].append([(value, index >= 8, 1)] * 4)
    maximum = qualified_cycles(observations)
    paired = {
        name: qualified_cycles([observations[a], observations[b]])
        for name, a, b in (("whole_vs_three", 0, 1), ("whole_vs_two", 2, 3), ("three_vs_two", 1, 3))
    }
    assert min(len(values) for values in maximum) >= 6
    assert choose_three_then_two(maximum, controlled, paired) == 3
    for calls in (16, 17, 20, 28, 32, 40, 42):
        slots = min(2 * calls, 64)
        order = two_pass_order(calls, slots)
        assert len(order) <= 2 * calls and all(0 <= choice < 4 for choice in order)
        assert min(Counter(order)[choice] for choice in (1, 3)) >= 7
    for heads in range(1, 33):
        for tokens in (1024, 4096, 9216, 14040, 16384):
            choices = three_then_two_candidates(heads, tokens, 128, 4, 132, kv_tokens=tokens + 4055)
            assert choices[0] == (heads,) and len(choices) <= 3
            assert len(set(choices)) == len(choices)
            assert all(sum(parts) == heads and min(parts) > 0 for parts in choices)
            if len(choices) > 1 and len(choices[1]) == 3:
                assert choices[1][0] == choices[1][-1] <= choices[1][1]
            if len(choices) == 3:
                assert len(choices[1]) == 3 and len(choices[2]) == 2
    for heads, tokens, world in (
        (12, 8335, 2),
        (12, 20623, 2),
        (12, 14518, 2),
        (12, 47286, 2),
        (10, 7800, 4),
        (10, 14040, 4),
        (14, 9248, 4),
        (14, 16416, 4),
    ):
        print(heads, tokens, candidates(heads, tokens, 128, world, 132))
    print("PASS conservative/noisy selection, exact coverage, odd eligibility, rank-independent balanced order")


if __name__ == "__main__":
    main()
