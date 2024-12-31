CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cner"


python run_ner_span.py --model_type=bert --model_name_or_path="C:/Users/29864/Desktop/BERT-NER-Pytorch/prev_trained_model/bert-base-chinese" --task_name="cner" --do_train --do_eval --do_adv --do_lower_case --loss_type=ce --data_dir="C:/Users/29864/Desktop/BERT-NER-Pytorch/datasets/cner/" --train_max_seq_length=128 --eval_max_seq_length=512 --per_gpu_train_batch_size=24 --per_gpu_eval_batch_size=24 --learning_rate=2e-5 --num_train_epochs=5.0 --logging_steps=-1 --save_steps=-1 --output_dir="C:/Users/29864/Desktop/BERT-NER-Pytorch/outputs/cner_output/" --overwrite_output_dir --seed=42


python run_ner_span.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_adv \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=2e-5 \
  --num_train_epochs=4.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42

