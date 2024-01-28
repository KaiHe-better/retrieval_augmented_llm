
# nohup python run.py --ID 0 --gpu 5 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  --dropout 0 >/dev/null 2>&1 &  
# nohup python run.py --ID 0 --gpu 7 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  --dropout 0.05 >/dev/null 2>&1 &  

# nohup python run.py --ID tt0 --gpu 5 --config llama2-7b_MedMCQA.yaml --dataset MedMCQA --seed 42 >/dev/null 2>&1 &  # acc 32.56, f1 14.33
# nohup python run.py --ID tt1 --gpu 7 --config llama2-7b_MedMCQA.yaml --dataset MedMCQA --seed 12312 >/dev/null 2>&1 & # acc 32.56, f1 14.33

# nohup python run.py --ID tt0 --gpu 5 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --seed 42 >/dev/null 2>&1 &   # acc 36.19, f1 34.53
# nohup python run.py --ID tt1 --gpu 7 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --seed 12312 >/dev/null 2>&1 &  # acc 36.19, f1 34.53

# --l2_coef 0 
# --lr 1e-4
# --nhead 8
# --num_layers 4
# --quantile_num 0.95
# --hierarchical_ratio 1.4


#  --retrieval_corpus_ids 0       acc 38.48 -> 40 	   
#  --retrieval_corpus_ids 0_1_2   acc 39.43 -> 41 	
# nohup python run.py --ID 1 --gpu 3 --config llama2-13b_USMLE_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --num_layers 3 >/dev/null 2>&1 &  # acc 39.43


# nohup python run.py --ID 0 --gpu 5 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  # MI_37.94
# nohup python run.py --ID 1 --gpu 7 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 &  # MI_38.73
# nohup python run.py --ID 2 --gpu 2 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 8e-5 >/dev/null 2>&1 &  # MI_38.96
# nohup python run.py --ID 3 --gpu 3 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 >/dev/null 2>&1 &  # MI_39.43

# --num_layers 4
# nohup python run.py --ID 0 --gpu 5 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --num_layers 2 >/dev/null 2>&1 &  # MI_39.28
# nohup python run.py --ID 1 --gpu 7 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --num_layers 3 >/dev/null 2>&1 &  # MI_39.98  （300）
# nohup python run.py --ID 2 --gpu 0 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --num_layers 4 >/dev/null 2>&1 &  # MI_39.43
# nohup python run.py --ID 2 --gpu 3 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --quantile_num 0.9 >/dev/null 2>&1 &  # MI_39.51


# nohup python run.py --ID 0_no_seed --gpu 0 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --num_layers 3  >/dev/null 2>&1 & # MI_38.18
# nohup python run.py --ID 1 --gpu 7 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --num_layers 3 --quantile_num 0.9 >/dev/null 2>&1 &  # MI_39.04
# nohup python run.py --ID 2 --gpu 0 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --num_layers 3 --loss_list kl_soft+kl_hard+mse --mse_weight 0.4 --soft_weight 0.3 --hard_weight 0.3 >/dev/null 2>&1 & # MI_37.71
# nohup python run.py --ID 3 --gpu 3 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 0_1_2 --lr 3e-4 --num_layers 3 --loss_list kl_soft+kl_hard --mse_weight 0.1 --soft_weight 0.3 --hard_weight 0.6 >/dev/null 2>&1 & # MI_38.02




# # acc 48.47 -> 50   
nohup python run.py --ID 0 --gpu 3,4,7 --config llama2-70b_USMLE_MI_RA.yaml --dataset USMLE  >/dev/null 2>&1 &  # MI_47.92
nohup python run.py --ID 2 --gpu 3,4,7 --config llama2-70b_USMLE_MI_RA.yaml --dataset USMLE --lr 3e-4 --num_layers 3  >/dev/null 2>&1 & 


# nohup python run.py --ID USMLE_multi_Q --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --multi_query True --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 --epoch 1 >/dev/null 2>&1 & 



nohup python run.py --ID MedMCQA_7_0 --gpu 5 --config llama2-7b_MedMCQA.yaml --dataset MedMCQA  >/dev/null 2>&1 &  # acc 32.56, f1 14.33  3.65 min
nohup python run.py --ID MedMCQA_7_1 --gpu 6 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA  >/dev/null 2>&1 & 

nohup python run.py --ID MedMCQA_13_0 --gpu 5 --config llama2-13b_MedMCQA.yaml --dataset MedMCQA  >/dev/null 2>&1 & # acc 35.79, f1 35.46  2.76 min
nohup python run.py --ID MedMCQA_13_1 --gpu 6 --config llama2-13b_MedMCQA_RA.yaml --dataset MedMCQA  >/dev/null 2>&1 & # acc 37.56, f1 35.88  52.85 min

nohup python run.py --ID HEADQA_7_0 --gpu 4 --config llama2-7b_HEADQA.yaml --dataset HEADQA  >/dev/null 2>&1 &  # acc 26.84, f1 15.92  1.55 min
nohup python run.py --ID HEADQA_7_1 --gpu 5 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA  >/dev/null 2>&1 & # acc 43.91, f1 43.42 26.88min

nohup python run.py --ID HEADQA_13_0 --gpu 4 --config llama2-13b_HEADQA.yaml --dataset HEADQA  >/dev/null 2>&1 &  # acc 48.18, f1 47.59  2.18 min
nohup python run.py --ID HEADQA_13_1 --gpu 5 --config llama2-13b_HEADQA_RA.yaml --dataset HEADQA  >/dev/null 2>&1 & # acc 51.46, f1 50.14  38.36 min

