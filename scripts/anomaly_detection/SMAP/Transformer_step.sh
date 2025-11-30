#!/bin/bash

# 设置工作目录为项目根目录
cd "$(dirname "$0")/../../../" || exit 1

export CUDA_VISIBLE_DEVICES=0

echo "当前工作目录: $(pwd)"
echo "开始运行Transformer_step异常检测实验..."

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len 96 \
  --step 1 \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \

  --anomaly_ratio 2.6 \
  --batch_size 32 \
  --train_epochs 30

echo "step = 1 实验完成"

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len 96 \
  --step 2 \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio 2.6 \
  --batch_size 32 \
  --train_epochs 30

echo "step = 2 实验完成"

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len 96 \
  --step 5 \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio 2.6 \
  --batch_size 32 \
  --train_epochs 30

echo "step = 5 实验完成"

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len 96 \
  --step 10 \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio 2.6 \
  --batch_size 32 \
  --train_epochs 30

echo "step = 10 实验完成"

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len 96 \
  --step 20 \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio 2.6 \
  --batch_size 32 \
  --train_epochs 30

echo "step = 20 实验完成"

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len 96 \
  --step 40 \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio 2.6 \
  --batch_size 32 \
  --train_epochs 30

echo "step = 40 实验完成"

python run_vibra.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./datasets/three_axis_vibra \
  --model_id SMAP \
  --model Transformer \
  --data SMAP \
  --features M \
  --seq_len 96 \
  --step 80 \
  --d_model 16 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 3 \
  --c_out 3 \
  --anomaly_ratio 2.6 \
  --batch_size 32 \
  --train_epochs 30

echo "step = 80 实验完成"


echo "所有Transformer异常检测实验运行完成！"

