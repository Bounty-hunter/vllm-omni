# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager

import torch


@contextmanager
def deterministic_cudnn():
    """Use deterministic convolutions and honor the caller's benchmark setting.

    Autotuning can select a different convolution algorithm on each cold
    start, even among deterministic algorithms.
    """
    old_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = old_deterministic
