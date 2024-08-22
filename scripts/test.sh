#! /bin/bash
export CUDA_VISIBLE_DEVICES='2'
export PYTHONPATH=/home/daiyf/daiyf/glucose-prediction/src  # {abs_path_to_glucose-prediction}/src
export data_path=/home/daiyf/daiyf/glucose-prediction/data/newdata  # {abs_path_to_glucose-prediction}/data/newdata

export test_sample='10,11,12,13,14,15,16'   # 测试样本集
export load_model=/home/daiyf/daiyf/glucose-prediction/src/our_result/CNN3.1-Linear10.0.pth
# 训练的最优模型参数存储路径，load下来做测试

python src/test.py \
  --file_path ${data_path} \
  --test_id_list ${test_sample} \
  --load_model ${load_model}