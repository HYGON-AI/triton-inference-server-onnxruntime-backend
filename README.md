<!--
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
-->
[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)
# ONNX Runtime Backend

HYGON ONNX Runtime inference backend for Triton Inference Server. It executes
ONNX models on HCU devices through the HIP runtime and the ONNX Runtime ROCm
execution provider, falling back to CPU when the ROCm execution provider cannot
be created.

- License: BSD-3-Clause, see [LICENSE](LICENSE)
- Third-party sources and their licenses: see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

