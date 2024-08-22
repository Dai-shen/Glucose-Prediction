#! /bin/bash
export CUDA_VISIBLE_DEVICES='1'
export PYTHONPATH=/home/daiyf/daiyf/glucose-prediction  # abs_path_to_glucose-prediciton
export load_path=/home/daiyf/daiyf/glp/data  # abs_path_to_original-data
export person_list='004,005,006,007'
export save_path=/home/daiyf/daiyf/glucose-prediction/data/newdata  # {abs_path_to_glucose-prediciton}/data/newdata
mkdir -p ${save_path}

python data/rawdata/preprocess.py \
  --load_path ${load_path} \
  --person_list ${person_list} \
  --save_path ${save_path}