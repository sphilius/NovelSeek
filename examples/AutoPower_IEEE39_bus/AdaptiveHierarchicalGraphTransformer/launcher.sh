#!/bin/bash
out_dir=$1
rseed=2024
ROOT=.
python $ROOT/experiment.py \
--config configs/test_senseflow_39.yaml \
--out_dir ${out_dir} \
--seed=${rseed} 