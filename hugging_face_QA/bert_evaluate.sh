python3.8 run_qa.py \
  --model_name_or_path ./fine_tune_mrc_squad \
  --validation_file ./data/json/VU_squad2.0_validate.json\
  --version_2_with_negative \
  --do_eval\
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./fine_tune_mrc_squad/
