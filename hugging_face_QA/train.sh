python3.8 run_seq2seq_qa.py \
  --model_name_or_path ./t5-base-Chinese \
  --train_file ./data/VU_squad2.0_hug.json\
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./fine_tune_seq2seq_squad/
