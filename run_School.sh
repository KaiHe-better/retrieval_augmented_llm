nohup python run.py --ID 1_USMLE --gpu 1 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0  >/dev/null 2>&1 &  #  acc 36.76
nohup python run.py --ID 2_USMLE --gpu 2 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0 --rewrite_num 1 --infer_retri_num 3   >/dev/null 2>&1 &  #  acc 36.14 
nohup python run.py --ID 3_USMLE --gpu 3 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0 --rewrite_num 2 --infer_retri_num 2   >/dev/null 2>&1 &  # test: acc 36.53

nohup python run.py --ID 1_USMLE --gpu 1 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  #  acc 38.88
nohup python run.py --ID 2_USMLE --gpu 2 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 3  >/dev/null 2>&1 &#  test: acc 40.93

nohup python run.py --ID 0_USMLE --gpu 1 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 &  #  acc 38.88
nohup python run.py --ID 0_USMLE --gpu 0 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 & # test: acc 40.77
nohup python run.py --ID 4_USMLE --gpu 1 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True --rewrite_num 2 --infer_retri_num 2 >/dev/null 2>&1 & # test: acc 40.53

nohup python run.py --ID 0 --gpu 0 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0   >/dev/null 2>&1 &  # MI_38.88
nohup python run.py --ID 1 --gpu 1 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1   >/dev/null 2>&1 &  # MI_38.02
nohup python run.py --ID 2 --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2   >/dev/null 2>&1 &  # MI_38.33

nohup python run.py --ID 0 --gpu 0 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0   --multi_query True --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 &  
nohup python run.py --ID 1 --gpu 1 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1  --multi_query True --rewrite_num 1 --infer_retri_num 3  >/dev/null 2>&1 &  
nohup python run.py --ID 2 --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2 --multi_query True --rewrite_num 1 --infer_retri_num 3   >/dev/null 2>&1 & 

# old multi_query
nohup python run.py --ID 4 --gpu 3 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1 --multi_query True  >/dev/null 2>&1 & # MI_37.63
nohup python run.py --ID 5 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  >/dev/null 2>&1 & # MI_38.18

====================================================================================================================================================================================================

nohup python run.py --ID 0_HEADQA --gpu 3 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0    >/dev/null 2>&1 &  # acc: 43.95
nohup python run.py --ID 1_HEADQA --gpu 0 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1    >/dev/null 2>&1 & # acc 46.46
nohup python run.py --ID 3_HEADQA --gpu 3 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2    >/dev/null 2>&1 &  # acc 46.32 

nohup python run.py --ID 4_HEADQA --gpu 4 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  --rewrite_num 1 --infer_retri_num 3  >/dev/null 2>&1 & # acc 45.84
nohup python run.py --ID 5_HEADQA --gpu 4 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  --rewrite_num 2 --infer_retri_num 2  >/dev/null 2>&1 & # acc 44.78

nohup python run.py --ID HEADQA_0 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & 
nohup python run.py --ID HEADQA_1 --gpu 4 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 4 --train_retri_num 3 >/dev/null 2>&1 & 

nohup python run.py --ID HEADQA_1 --gpu 1 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1  >/dev/null 2>&1 & # MI_45.44
nohup python run.py --ID HEADQA_2 --gpu 2 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 & # MI_45.33

nohup python run.py --ID 3 --gpu 3 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1 --multi_query True  --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 & # MI_45.33
nohup python run.py --ID 4 --gpu 4 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 & # MI_45.48

====================================================================================================================================================================================================

nohup python run.py --ID MedMCQA_0 --gpu 0 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0  >/dev/null 2>&1 & # 36.17
nohup python run.py --ID MedMCQA_1 --gpu 1 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 & # acc 54.77
nohup python run.py --ID MedMCQA_2 --gpu 2 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 & # acc 54.51

nohup python run.py --ID MedMCQA_1 --gpu 1 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  >/dev/null 2>&1 &
nohup python run.py --ID MedMCQA_2 --gpu 2 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --hierarchical_ratio 2   >/dev/null 2>&1 &
nohup python run.py --ID MedMCQA_3 --gpu 3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --hierarchical_ratio 1.7 >/dev/null 2>&1 &
nohup python run.py --ID MedMCQA_4 --gpu 5 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --if_hierarchical_retrieval False >/dev/null 2>&1 &

nohup python run.py --ID MedMCQA_3 --gpu 3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  --retrieval_corpus_ids 0 --multi_query True  --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 &



====================================================================================================================================================================================================

nohup  python run.py --ID retwrite_0 --dataset USMLE --config llama2-7b_USMLE_RA.yaml --rewrite_num 2 --test_batch_size 1 >/dev/null 2>&1 &
nohup  python run.py --ID retwrite_1 --dataset MedMCQA --gpu 7  --config llama2-7b_MedMCQA_RA.yaml --rewrite_num 2 --test_batch_size 1 >/dev/null 2>&1 &
nohup  python run.py --ID retwrite_2 --dataset HEADQA --config llama2-7b_HEADQA_RA.yaml --rewrite_num 2 --test_batch_size 1 >/dev/null 2>&1 &
nohup  python run.py --ID retwrite_2 --dataset MMLU --config llama2-7b_MMLU_RA.yaml --rewrite_num 2 --test_batch_size 1 >/dev/null 2>&1 &

