#!/usr/bin/env bash
# Class-imbalance experiments on DenseNet-121: repeated baselines,
# loss reweighting and oversampling of rare answers to Q8-Q10.
# Finished runs are skipped, so the script can be stopped and restarted.
set -euo pipefail

ARCH=densenet121.tv_in1k
OUT=models/imbalance

run() {
    local name=$1; shift
    if [[ -f "$OUT/${name}_run.json" ]]; then
        echo "skip $name (finished)"
        return
    fi
    python -m scripts.train --arch "$ARCH" --out-dir "$OUT" --run-name "$name" "$@"
}

# Every configuration is repeated over the same ten seeds, so the effect of an
# intervention can be compared with the seed-to-seed spread of the baseline.
for seed in 101 202 303 404 505 606 707 808 909 1010; do
    run "imbalance_baseline_s${seed}"    --seed "$seed"
    run "imbalance_classweight_s${seed}" --seed "$seed" --class-weights
    run "imbalance_oversample_s${seed}"  --seed "$seed" --oversample
done
