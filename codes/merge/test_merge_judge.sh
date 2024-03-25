SEED=42
python test_merge.py \
    --data_dir ../../datasets/example_data \
    --train_filename merged_train.json \
    --data_test_dir ../../datasets/example_data \
    --data_merged_file ../llm_inference/merged.json \
    --model_name_or_path ../../hf-models/roberta-large \
    --input_format typed_entity_marker_punct \
    --seed $SEED \
    --test_batch_size 4 \
    --checkpoint_filepath ../traditional_re/wandb/run-xxxx/files/checkpoints/epoch_xxxx/model.pt \
    --temp_output_merged_filepath ./temp_output_merged.json \

# checkpoint_filepath refers to the selected checkpoints filepath


python test_judge.py \
    --train_path ../../datasets/example_data/train.json \
    --test_path ./temp_output_merged.json \
    --output_success ./ \
    --output_nores ./ \
    --prompt instruct_schema \
    --k 4 \
    --llm_modelpath ../../hf-models/zephyr-7b-alpha \


python merge_cal_res.py