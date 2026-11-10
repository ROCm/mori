# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Build from the repository root; no GPU or model weights are needed.
ARG BASE_IMAGE=lmsysorg/sglang-rocm@sha256:9b82c0b334618afe6182a542fe4adf3694b9b059edfeb41b997a22f2fd4c0a86
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

WORKDIR /opt/umbp-demo
COPY examples/umbp/sglang/ /opt/umbp-demo/
COPY examples/umbp/sglang_linker_smoke.py /opt/umbp-demo/sglang_linker_smoke.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
RUN python3 prepare_image.py --base-image "$BASE_IMAGE" \
    && python3 preflight.py \
    && python3 recipe/test_recipe.py \
    && python3 test_demo.py
ENTRYPOINT ["python3", "/opt/umbp-demo/container_entry.py"]
CMD ["check"]
