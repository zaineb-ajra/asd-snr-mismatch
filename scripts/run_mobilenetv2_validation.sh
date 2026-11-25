#!/usr/bin/env bash
# MobileNetV2 code validation: fidelity check on DCASE 2022 (5 types x 5 seeds,
# validation_split 0.1), control run ToyCar seed 0 with validation_split 0.0, and
# internal controls on the study data (0 dB, seed 0). Four queues, two per GPU;
# the first run of each DCASE machine type is in a single queue so that its
# features are computed once. Finished runs are skipped.
#
# Usage: scripts/run_mobilenetv2_validation.sh <gpu_a> <gpu_b>   (in tmux)
set -u
cd "$(dirname "$0")/.."
GPU_A=${1:?gpu a}
GPU_B=${2:?gpu b}
LOG=$(.venv/bin/python -c "from noisemismatch.config import load_paths; print(load_paths().output_root)")/verification/mobilenetv2/logs
mkdir -p "$LOG"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=".venv/bin/python -m"

fid() { # gpu type seed [validation_split]
  CUDA_VISIBLE_DEVICES=$1 nice -n 5 $PY noisemismatch.verification_dcase2022 \
    --types "$2" --seeds "$3" --validation-split "${4:-0.1}"
}
controls() { # gpu
  CUDA_VISIBLE_DEVICES=$1 nice -n 5 $PY noisemismatch.controls --model mobilenetv2 --train-snr 0
}
queue() { # name, then commands separated by ';'
  local name=$1; shift
  ( for job in "$@"; do
      echo "$(date '+%F %T') start: $job"
      eval "$job"; echo "$(date '+%F %T') exit $?: $job"
    done ) > "$LOG/$name.log" 2>&1
  echo done > "$LOG/$name.done"
}

queue q1 "controls $GPU_A" "fid $GPU_A ToyCar 0 0.0" "fid $GPU_A ToyCar 1" "fid $GPU_A ToyCar 3" &
queue q2 "fid $GPU_A ToyTrain 0" "fid $GPU_A ToyTrain 1" "fid $GPU_A ToyTrain 2" \
  "fid $GPU_A ToyTrain 3" "fid $GPU_A ToyTrain 4" "fid $GPU_A ToyCar 4" "fid $GPU_A slider 3" &
queue q3 "fid $GPU_B fan 0" "fid $GPU_B fan 1" "fid $GPU_B fan 2" "fid $GPU_B fan 3" \
  "fid $GPU_B fan 4" "fid $GPU_B slider 0" "fid $GPU_B slider 1" &
queue q4 "fid $GPU_B valve 0" "fid $GPU_B valve 1" "fid $GPU_B valve 2" "fid $GPU_B valve 3" \
  "fid $GPU_B valve 4" "fid $GPU_B ToyCar 2" "fid $GPU_B slider 2" "fid $GPU_B slider 4" &
wait
# Aggregation over every run (all runs are done, nothing is trained).
$PY noisemismatch.verification_dcase2022 > "$LOG/aggregate.log" 2>&1
echo $? > "$LOG/all.exitcode"
