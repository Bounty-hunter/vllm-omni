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
    metadata_spec,
    qualified_cycles,
    symmetric_candidates,
    tensor_spec,
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
    assert choose([[100] * 10, [99] * 10]) == 0
    assert choose([[100] * 5, [70] * 5]) == 0
    assert choose([[100] * 10, [80] * 10, [90] * 10]) == 1
    assert choose([[100] * 10, [110] * 10]) == 0
    assert choose([[100] * 10, [50, 150] * 5]) == 0
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
