#!/usr/bin/env bash
# Architecture benchmark: identical training setup for every backbone.
set -euo pipefail

ARCHS=(
    efficientnet_b0.ra_in1k
    resnet18.a1_in1k resnet50.a1_in1k resnet152.a1_in1k
    densenet121.tv_in1k densenet169.tv_in1k densenet201.tv_in1k densenet161.tv_in1k
    convnext_tiny.fb_in1k convnext_small.fb_in1k convnext_base.fb_in1k
    maxvit_tiny_tf_224.in1k maxvit_small_tf_224.in1k maxvit_base_tf_224.in1k
)

for arch in "${ARCHS[@]}"; do
    python -m scripts.train --arch "$arch"
done
