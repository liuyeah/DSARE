prefix="../../datasets/example_data/llm_generated" 
file_end=".json"
python automodel_DA.py \
    --demo_path ../../datasets/example_data/train.json \
    --auto_modelpath ../../hf-models/zephyr-7b-alpha \
    --output_dir $prefix$file_end \
    --dataset tacred \
    --k 3;
# dataset: tacred, tacrev, retacred

python merge_data.py