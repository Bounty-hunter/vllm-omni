# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the performance gate, not synthetic GPU performance claims."""

from auto_policy_cases import TYPICAL_CASES, WAN_CASES
from auto_policy_statistics import acceptance, paired_latency


def main():
    stable = [(100.0, 100.0)] * 12
    assert acceptance([(100.0, 98.0)] * 12, stable)["status"] == "pass"
    assert acceptance([(100.0, 100.9)] * 12, stable)["status"] == "pass"
    assert acceptance([(100.0, 102.0)] * 12, stable)["status"] == "regressed"
    noisy = [(100.0, value) for value in (95, 106) * 6]
    assert acceptance(noisy, stable)["status"] == "inconclusive"
    assert acceptance(stable, noisy)["status"] == "inconclusive"
    slow = stable[:-1] + [(100.0, 130.0)]
    result = paired_latency(slow)
    assert result["automatic_mean_us"] == 102.5 and result["status"] != "pass"
    for bad in ([], [(100.0, 100.0)] * 5, [(100.0, float("nan"))] * 6, [(100.0, 0.0)] * 6):
        try:
            paired_latency(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid measurements were accepted")
    assert len(TYPICAL_CASES) == 11 and len(WAN_CASES) == 2
    for case in TYPICAL_CASES + WAN_CASES:
        assert case.tokens % 4 == 0 and case.heads % 4 == 0
        assert 0 <= case.padding < case.tokens
        assert case.joint_kv >= case.joint_q
        assert not any("partition" in field or "selected" in field for field in case.__dataclass_fields__)
    print("PASS paired 1% gate, uncertainty, retained slow samples and partition-free input inventory")


if __name__ == "__main__":
    main()
