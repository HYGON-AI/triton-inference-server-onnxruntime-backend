# Third-Party Notices

This repository contains the following third-party source. The original
copyright and license notices are preserved in the files and are not replaced
or supplemented with HYGON headers.

## ONNX Runtime C API header

- Project: ONNX Runtime
- Repository: https://github.com/microsoft/onnxruntime
- Version: `v1.19.2` (fixed release tag)
- Source: https://github.com/microsoft/onnxruntime/blob/v1.19.2/include/onnxruntime/core/session/onnxruntime_c_api.h
- Upstream file: https://raw.githubusercontent.com/microsoft/onnxruntime/v1.19.2/include/onnxruntime/core/session/onnxruntime_c_api.h
- SHA-256: `f4047359e0dbf2078fff0e88bfb806de3c2b8891a895ac0dffdc3dfcb8bb489b`
- API version: `ORT_API_VERSION` 19
- Copyright: Copyright (c) Microsoft Corporation. All rights reserved.
- License: MIT
- Local path: `include/onnxruntime_c_api.h`
- Modifications: none. The file is carried verbatim as the ONNX Runtime C API
  header required to build the backend.
- Verification: the committed copy is byte-identical to the upstream `v1.19.2`
  header (SHA-256 after normalizing line endings to LF; the working-tree
  checkout may use CRLF on Windows). The original import commit is not recorded
  in this repository; `v1.19.2` is the fixed upstream version this copy is
  verified against, and `CMakeLists.txt` requires ONNX Runtime 1.19.2 and
  downloads the header from the matching `v${ORT_VERSION}` tag when the local
  copy is missing or outdated.
