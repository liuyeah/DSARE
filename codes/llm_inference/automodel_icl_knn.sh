python automodel_ICL_knn.py \
    --train_path ../../datasets/example_data/train.json \
    --test_path ../traditional_re/wandb/run-xxxx/files/checkpoints/epoch_xxxx/tacred_augment_data.json \
    --output_success ./ \
    --output_nores ./ \
    --prompt instruct_schema \
    --k 8 \
    --auto_modelpath ../../hf-models/zephyr-7b-alpha \

python merge4merge.py