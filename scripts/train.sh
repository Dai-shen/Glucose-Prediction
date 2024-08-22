#! /bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/daiyf/daiyf/glucose-prediction/src # abs_path_to_glucose-prediction/src
export data_path=/home/daiyf/daiyf/glucose-prediction/data/newdata  # {abs_path_to_glucose-prediction}/data/newdata
export train_sample='02'  # 训练样本集
export model_name=DCNN

export save_model=/home/daiyf/daiyf/glucose-prediction/src/${model_name}_result  # 模型保存路径
mkdir -p ${save_model}

export log_dir=/home/daiyf/daiyf/glucose-prediction/src   # 日志文件
echo ${log_dir}/train.log

nohup python -u src/train_main.py \
  --file_path ${data_path} \
  --train_id_list ${train_sample} \
  --save_model ${save_model} \
  --model_name ${model_name} \
  --epochs 100 \
  > ${log_dir}/train.log 2>&1 &