# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for conservative selection and balanced natural-call sampling."""

from collections import Counter

from lazy_head_policy import candidate_index, candidates, choose


def main():
    for heads in (10, 12, 14):
        for tokens in (288, 1056, 1344, 2048, 7800, 8128, 9248, 14040, 16416, 20623, 26806, 47286):
            choices = candidates(heads, tokens, 128, 4 if heads != 12 else 2, 132)
            assert choices[0] == (heads,) and len(choices) <= 3
            assert len(set(choices)) == len(choices)
            assert all(sum(widths) == heads and min(widths) > 0 for widths in choices)
    assert any(any(width % 2 for width in widths) for widths in candidates(14, 9248, 128, 4, 132)[1:])
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
