#!/bin/bash

# 设置工作目录为项目根目录
cd "$(dirname "$0")/../../../" || exit 1

export CUDA_VISIBLE_DEVICES=0

echo "当前工作目录: $(pwd)"
echo "开始运行Transformer_thres异常检测实验..."

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len  \
  --step  \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio 1 \
  --batch_size 32 \
  --train_epochs 30

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len  \
  --step  \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio  \
  --batch_size 32 \
  --train_epochs 30

echo "threshold =  实验完成"

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len  \
  --step  \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio  \
  --batch_size 32 \
  --train_epochs 30

echo "threshold =  实验完成"

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len  \
  --step  \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio  \
  --batch_size 32 \
  --train_epochs 30

echo "threshold =  实验完成"

echo "所有Transformer异常检测实验运行完成！"

