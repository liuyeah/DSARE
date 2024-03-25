cd data_augmentation/
bash data_augment.sh
cd ..

cd traditional_re/
bash ly_fewshot_scripts/run_roberta.sh
cd ..

cd demonstration/
bash recall.sh
cd ..

cd llm_inference/
bash automodel_icl_knn.sh
cd ..

cd merge/
bash test_merge_judge.sh
cd ..
