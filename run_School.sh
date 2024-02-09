
# nohup  python run.py --ID retwrite_0 --dataset USMLE --config llama2-7b_USMLE_RA.yaml --rewrite_num 2 --test_batch_size 1 >/dev/null 2>&1 &
# nohup  python run.py --ID retwrite_1 --dataset MedMCQA --gpu 7  --config llama2-7b_MedMCQA_RA.yaml --rewrite_num 2 --test_batch_size 1 >/dev/null 2>&1 &
# nohup  python run.py --ID retwrite_2 --dataset HEADQA --config llama2-7b_HEADQA_RA.yaml --rewrite_num 2 --test_batch_size 1 >/dev/null 2>&1 &
# nohup  python run.py --ID retwrite_2 --dataset MMLU --config llama2-7b_MMLU_RA.yaml --rewrite_num 2 --test_batch_size 1 >/dev/null 2>&1 &

====================================================================================================================================================================================================

# nohup python run.py --ID 1_USMLE --gpu 1 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0  >/dev/null 2>&1 &  #  acc 36.76
# nohup python run.py --ID 2_USMLE --gpu 2 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0 --rewrite_num 1 --infer_retri_num 3   >/dev/null 2>&1 &  #  acc 36.14 
# nohup python run.py --ID 3_USMLE --gpu 3 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0 --rewrite_num 2 --infer_retri_num 2   >/dev/null 2>&1 &  # test: acc 36.53

# nohup python run.py --ID 1_USMLE --gpu 1 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  #  acc 38.88
# nohup python run.py --ID 2_USMLE --gpu 2 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 3  >/dev/null 2>&1 &#  test: acc 40.93

# nohup python run.py --ID 0_USMLE --gpu 1 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 &  #  acc 38.88
# nohup python run.py --ID 0_USMLE --gpu 0 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 & # test: acc 40.77
# nohup python run.py --ID 4_USMLE --gpu 1 --config llama2-7b_USMLE_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True --rewrite_num 2 --infer_retri_num 2 >/dev/null 2>&1 & # test: acc 40.53

# nohup python run.py --ID USMLE_7_0 --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0   >/dev/null 2>&1 & 
# nohup python run.py --ID 0 --gpu 0 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0   >/dev/null 2>&1 &  # MI_38.88
# nohup python run.py --ID 1 --gpu 1 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1   >/dev/null 2>&1 &  # MI_38.02
# nohup python run.py --ID 2 --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2   >/dev/null 2>&1 &  # MI_38.33

# nohup python run.py --ID 0 --gpu 0 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0   --multi_query True --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 &  
# nohup python run.py --ID 1 --gpu 1 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1  --multi_query True --rewrite_num 1 --infer_retri_num 3  >/dev/null 2>&1 &  
# nohup python run.py --ID 2 --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2 --multi_query True --rewrite_num 1 --infer_retri_num 3   >/dev/null 2>&1 & 

# # old multi_query
# nohup python run.py --ID 4 --gpu 3 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1 --multi_query True  >/dev/null 2>&1 & # MI_37.63
# nohup python run.py --ID 5 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  >/dev/null 2>&1 & # MI_38.18

====================================================================================================================================================================================================

# nohup python run.py --ID 0_HEADQA --gpu 3 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0    >/dev/null 2>&1 &  # acc: 43.95
# nohup python run.py --ID 1_HEADQA --gpu 0 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1    >/dev/null 2>&1 & # acc 46.46
# nohup python run.py --ID 3_HEADQA --gpu 3 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2    >/dev/null 2>&1 &  # acc 46.32 

# nohup python run.py --ID 4_HEADQA --gpu 4 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  --rewrite_num 1 --infer_retri_num 3  >/dev/null 2>&1 & # acc 45.84
# nohup python run.py --ID 5_HEADQA --gpu 4 --config llama2-7b_HEADQA_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  --rewrite_num 2 --infer_retri_num 2  >/dev/null 2>&1 & # acc 44.78

# nohup python run.py --ID HEADQA_1 --gpu 1 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1  >/dev/null 2>&1 & # MI_45.44
# nohup python run.py --ID HEADQA_2 --gpu 2 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 & # MI_45.33

# nohup python run.py --ID 3 --gpu 3 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1 --multi_query True  --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 & # MI_45.33
# nohup python run.py --ID 4 --gpu 4 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 & # MI_45.48


# # > 43.91 
# nohup python run.py --ID HEADQA_0 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # best_step:100, best_performce: 42.56
# nohup python run.py --ID HEADQA_1 --gpu 4 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 4 --train_retri_num 3 >/dev/null 2>&1 & # best_step:100, best_performce: 43.07

# nohup python run.py --ID HEADQA_5 --gpu 5 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA >/dev/null 2>&1 & # MI_44.57
# nohup python run.py --ID HEADQA_0 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --quantile_num 0.95 >/dev/null 2>&1 &  # MI_43.51
# nohup python run.py --ID HEADQA_1 --gpu 1 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --quantile_num 0.8 >/dev/null 2>&1 &   # MI_44.13
# nohup python run.py --ID HEADQA_2 --gpu 2 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --if_hierarchical_retrieval True >/dev/null 2>&1 &  # MI_44.57
# nohup python run.py --ID HEADQA_3 --gpu 3 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --quantile_num 0.9  --if_hierarchical_retrieval True >/dev/null 2>&1 & # MI_44.46
# nohup python run.py --ID HEADQA_4 --gpu 4 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --quantile_num 0.8  --if_hierarchical_retrieval True >/dev/null 2>&1 & # MI_44.57
# nohup python run.py --ID HEADQA_5 --gpu 5 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3  --quantile_num 0.85 >/dev/null 2>&1 & # 42.43 

# nohup python run.py --ID HEADQA_0 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 2 --infer_retri_num 2 --train_retri_num 2 >/dev/null 2>&1 & # MI_41.79
# nohup python run.py --ID HEADQA_0 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null --quantile_num 0.5  2>&1 & # MI_43.33
# nohup python run.py --ID HEADQA_1 --gpu 1 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3  --hierarchical_ratio 2 --quantile_num 0.7 >/dev/null 2>&1 & # MI_42.92
# nohup python run.py --ID HEADQA_2 --gpu 2 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3  --quantile_num 0.8 >/dev/null 2>&1 & # MI_43.47
# nohup python run.py --ID HEADQA_3 --gpu 3 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3  --hierarchical_ratio 1.2 >/dev/null 2>&1 & # MI_42.3
# nohup python run.py --ID HEADQA_4 --gpu 4 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 4 --train_retri_num 3  --hierarchical_ratio 1.6 >/dev/null 2>&1 & # MI_42.92
# nohup python run.py --ID HEADQA_5 --gpu 5 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --multi_query True  --rewrite_num 1 --infer_retri_num 4 --train_retri_num 4  --hierarchical_ratio 1.6 >/dev/null 2>&1 & # MI_43.76


# nohup python run.py --ID HEADQA_00 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --hierarchical_ratio 1.4 >/dev/null 2>&1 &   #  best_step:100, best_performce: 44.57
# nohup python run.py --ID HEADQA_0 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --hierarchical_ratio 1.2 >/dev/null 2>&1 & 
# nohup python run.py --ID HEADQA_1 --gpu 1 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --hierarchical_ratio 1.6 >/dev/null 2>&1 & 
# nohup python run.py --ID HEADQA_2 --gpu 2 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --hierarchical_ratio 1.8 >/dev/null 2>&1 & 
# nohup python run.py --ID HEADQA_3 --gpu 3 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --hierarchical_ratio 2.0 >/dev/null 2>&1 & 

====================================================================================================================================================================================================
# > 36.17

# nohup python run.py --ID MedMCQA_0 --gpu 0 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0  >/dev/null 2>&1 & # 36.17
# nohup python run.py --ID MedMCQA_1 --gpu 1 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 & # acc 54.77
# nohup python run.py --ID MedMCQA_2 --gpu 2 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 & # acc 54.51

# nohup python run.py --ID MedMCQA_1 --gpu 1 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --hierarchical_ratio 1.4 >/dev/null 2>&1 &  # best_step:700, best_performce: 36.41
# nohup python run.py --ID MedMCQA_2 --gpu 2 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --hierarchical_ratio 2   >/dev/null 2>&1 &  # best_step:900,  best_performce: 36.15
# nohup python run.py --ID MedMCQA_3 --gpu 3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --hierarchical_ratio 1.7 >/dev/null 2>&1 &  # best_step:2100, best_performce: 36.19
# nohup python run.py --ID MedMCQA_4 --gpu 5 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --if_hierarchical_retrieval False >/dev/null 2>&1 &  # 

# nohup python run.py --ID MedMCQA_3 --gpu 3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  --retrieval_corpus_ids 0 --multi_query True  --rewrite_num 1 --infer_retri_num 3 >/dev/null 2>&1 &


# nohup python run.py --ID MedMCQA_0 --gpu 4 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA  >/dev/null 2>&1 & # acc 35.79, f1 33.49
# nohup python run.py --ID MedMCQA_0 --gpu 4 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --hierarchical_ratio 2  >/dev/null 2>&1 & #  acc 35.79
# nohup python run.py --ID MedMCQA_0 --gpu 4 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --hierarchical_ratio 1.2 >/dev/null 2>&1 & # acc 36.17

# nohup python run.py --ID MedMCQA_1 --gpu 1 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA   >/dev/null 2>&1 &   # MI_36.41
# nohup python run.py --ID MedMCQA_2 --gpu 2 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  --hierarchical_ratio 1.2 >/dev/null 2>&1 & # MI_35.57
# nohup python run.py --ID MedMCQA_3 --gpu 3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  --hierarchical_ratio 1.5 >/dev/null 2>&1 & # MI_36.36
# nohup python run.py --ID MedMCQA_4 --gpu 4 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  --num_layers 2 >/dev/null 2>&1 & # MI_36.43
# nohup python run.py --ID MedMCQA_5 --gpu 5 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA  --num_layers 3 >/dev/null 2>&1 & # MI_36.77

=
# nohup python run.py --ID USMLE_7_0 --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  --retrieval_corpus_ids 0    >/dev/null 2>&1 &  # MI_38.88
# nohup python run.py --ID USMLE_7_1 --gpu 1 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  --retrieval_corpus_ids 0    --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_37.55
# nohup python run.py --ID USMLE_7_1_1 --gpu 3 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  --retrieval_corpus_ids 0    --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_38.88
# nohup python run.py --ID USMLE_7_2 --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  # MI_38.02
# nohup python run.py --ID USMLE_7_3 --gpu 3 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_40.06
# nohup python run.py --ID USMLE_7_3 --gpu 3 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_40.14
# nohup python run.py --ID USMLE_7_3_1 --gpu 3 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE  --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_40.06


===================================================================================================================================================================================================
====================================================================================================================================================================================================
====================================================================================================================================================================================================


## 跑错了， 7B 跑成layer 1了， 应该跑layer 3
# nohup python run.py --ID MedMCQA_7_0 --gpu 2 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0    >/dev/null 2>&1 &  # MI_36.77
# nohup python run.py --ID MedMCQA_7_1   --gpu 0 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0    --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_36.34
# nohup python run.py --ID MedMCQA_7_1_1 --gpu 4 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0    --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_36.0
# nohup python run.py --ID MedMCQA_7_2 --gpu 5 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  # MI_50.47
# nohup python run.py --ID MedMCQA_7_3 --gpu 3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_47.72
# nohup python run.py --ID MedMCQA_7_3_1 --gpu 0 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_49.25

# --num_layers 3，  对比上面的 7B
# nohup python run.py --ID MedMCQA_7_0 --gpu 3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0     --num_layers 3 >/dev/null 2>&1 &  # MI_36.77
# nohup python run.py --ID MedMCQA_7_1   --gpu 4 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0   --num_layers 3  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_36.46
# nohup python run.py --ID MedMCQA_7_1_1 --gpu 5 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0   --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 &  # MI_36.48
# nohup python run.py --ID MedMCQA_7_2 --gpu 0 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1   --num_layers 3 >/dev/null 2>&1 &  # 51.61
# nohup python run.py --ID MedMCQA_7_3 --gpu 1 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1   --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 &  # MI_49.61
# nohup python run.py --ID MedMCQA_7_3_1 --gpu 2,3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1 --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_51.64




# layer_3
# nohup python run.py --ID MedMCQA_13_0 --gpu 5 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0 --num_layers 3   >/dev/null 2>&1 &  # MI_37.08
# nohup python run.py --ID MedMCQA_13_1 --gpu 6 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0  --num_layers 3  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_36.55 
# nohup python run.py --ID MedMCQA_13_1_1 --gpu 7 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0  --num_layers 3  --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 &  # MI_37.13
# nohup python run.py --ID MedMCQA_13_2 --gpu 4,5 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1 --num_layers 3 >/dev/null 2>&1 &  # MI_51.52
# nohup python run.py --ID MedMCQA_13_3 --gpu 0,1 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1  --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_52.45
# nohup python run.py --ID MedMCQA_13_3—1 --gpu 2,3 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1  --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_53.29


# layer_1 no better
# nohup python run.py --ID MedMCQA_13_0 --gpu 0,1 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0    >/dev/null 2>&1 &  
# nohup python run.py --ID MedMCQA_13_1 --gpu 2,3 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0    --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & 
# nohup python run.py --ID MedMCQA_13_1_1 --gpu 4,5 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0    --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 &  
# nohup python run.py --ID MedMCQA_13_2 --gpu 4,5 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  # MI_51.52
# nohup python run.py --ID MedMCQA_13_3 --gpu 3 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_48.82
# nohup python run.py --ID MedMCQA_13_3—1 --gpu 0,1,2 --config llama2-13b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_52.21



==================================================================================================================================================================================================
====================================================================================================================================================================================================
====================================================================================================================================================================================================


# nohup python run.py --ID HEADQA_7_0 --gpu 1 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0    >/dev/null 2>&1 &  # MI_44.57
# nohup python run.py --ID HEADQA_7_1 --gpu 2 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0    --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_43.33
# nohup python run.py --ID HEADQA_7_2 --gpu 3 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  # MI_45.66
# nohup python run.py --ID HEADQA_7_3 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_45.81
# nohup python run.py --ID HEADQA_7_1_1 --gpu 5 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0    --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_41.90
# nohup python run.py --ID HEADQA_7_3_1 --gpu 5 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1  --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & # MI_45.73

# only 100
# nohup python run.py --ID HEADQA_13_0 --gpu 0_1 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --lr 1e-4 --num_layers 1   >/dev/null 2>&1 &  #MI_51.24
# nohup python run.py --ID HEADQA_13_1 --gpu 2_3 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --lr 1e-4 --num_layers 1 --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 &   # MI_50.44
# nohup python run.py --ID HEADQA_13_2 --gpu 4_5 `--config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --lr 1e-4 --num_layers 1 >/dev/null 2>&1 &  # MI_51.31

# nohup python run.py --ID HEADQA_13_0 --gpu 0,1 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --lr 3e-4 --num_layers 3  >/dev/null 2>&1 &  # MI_50.84
# nohup python run.py --ID HEADQA_13_1 --gpu 2,3 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --lr 3e-4 --num_layers 3  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & # MI_50.84
# nohup python run.py --ID HEADQA_13_2 --gpu 4,5 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --lr 3e-4 --num_layers 3 >/dev/null 2>&1 &  # MI_51.53
# nohup python run.py --ID HEADQA_13_3 --gpu 5 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --lr 3e-4 --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 &  # MI_50.22
# nohup python run.py --ID HEADQA_13_1_1 --gpu 6 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --lr 3e-4 --num_layers 3  --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 &  # MI_51.06
# nohup python run.py --ID HEADQA_13_3_1 --gpu 7 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1  --lr 3e-4 --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 &  # MI_51.46



# --quantile_num 1
# nohup python run.py --ID HEADQA_13_0 --gpu 0,1 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --quantile_num 1 --lr 3e-4 --num_layers 3  >/dev/null 2>&1 &  MI_49.74
# nohup python run.py --ID HEADQA_13_1 --gpu 2,3 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --quantile_num 1 --lr 3e-4 --num_layers 3  --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & MI_49.38
# nohup python run.py --ID HEADQA_13_2 --gpu 4,5 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --quantile_num 1 --lr 3e-4 --num_layers 3 >/dev/null 2>&1 &  MI_50.0
# nohup python run.py --ID HEADQA_13_3 --gpu 0,1,2 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --quantile_num 1 --lr 3e-4 --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 &  
# nohup python run.py --ID HEADQA_13_1_1 --gpu 3,4,5 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0 --quantile_num 1 --lr 3e-4 --num_layers 3  --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 &  
# nohup python run.py --ID HEADQA_13_3_1 --gpu XXX --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --quantile_num 1 --lr 3e-4 --num_layers 3 --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 &  


# --quantile_num 0.85
# nohup python run.py --ID HEADQA_13_1 --gpu 4,5,6 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --quantile_num 0.85 --lr 1e-4 --num_layers 1  --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & MI_50.95
# nohup python run.py --ID HEADQA_13_1_1 --gpu XXX --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0 --quantile_num 0.85 --lr 1e-4 --num_layers 1  --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 &  MI_50.0
# nohup python run.py --ID HEADQA_13_3_1 --gpu 0,1 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --quantile_num 0.85 --lr 1e-4 --num_layers 1 --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 


==================================================================================================================================================================================================



# --quantile_num 0.85 --lr 3e-4 --num_layers 3 
# nohup python run.py --ID HEADQA_13_0 --gpu 2,5 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --quantile_num 0.85 --lr 3e-4 --num_layers 3  >/dev/null 2>&1 &  MI_51.75

# nohup python run.py --ID HEADQA_13_1 --gpu 0,1  --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --quantile_num 0.85 --lr 3e-4 --num_layers 3  --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 & 
# nohup python run.py --ID HEADQA_13_2 --gpu 2,3  --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --quantile_num 0.85 --lr 3e-4 --num_layers 3  --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 

# nohup python run.py --ID HEADQA_13_2 --gpu 0,1 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --quantile_num 0.85 --lr 3e-4 --num_layers 3 >/dev/null 2>&1 &  MI_52.26
# nohup python run.py --ID HEADQA_13_3 --gpu 3,4 --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0_1 --quantile_num 0.85 --lr 3e-4 --num_layers 3 --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 3 --train_retri_num 3 >/dev/null 2>&1 &  MI_51.46 
# nohup python run.py --ID HEADQA_13_5 --gpu XXX  --config llama2-13b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --quantile_num 0.85 --lr 3e-4 --num_layers 3  --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 




==================================================================================================================================================================================================
====================================================================================================================================================================================================
====================================================================================================================================================================================================


# nohup python run.py --ID USMLE_7_0 --gpu 2 --config chatGPT_USMLE.yaml --dataset USMLE --retrieval_corpus_ids 0 --epoch 1   >/dev/null 2>&1 &  
# nohup python run.py --ID USMLE_7_1 --gpu 3 --config chatGPT_USMLE_RA.yaml --dataset USMLE --retrieval_corpus_ids 0  --epoch 1  >/dev/null 2>&1 &  
# nohup python run.py --ID MedMCQA_7_0 --gpu 4 --config chatGPT_MedMCQA.yaml --dataset MedMCQA --retrieval_corpus_ids 0 --epoch 1   >/dev/null 2>&1 &  
# nohup python run.py --ID MedMCQA_7_1 --gpu 5 --config chatGPT_MedMCQA_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 0  --epoch 1  >/dev/null 2>&1 &  
# nohup python run.py --ID HEADQA_7_0 --gpu 6 --config chatGPT_HEADQA.yaml --dataset HEADQA --retrieval_corpus_ids 0  --epoch 1  >/dev/null 2>&1 &  
# nohup python run.py --ID HEADQA_7_1 --gpu 7 --config chatGPT_HEADQA_RA.yaml --dataset HEADQA --retrieval_corpus_ids 0 --epoch 1   >/dev/null 2>&1 &  



# nohup python run.py --ID USMLE_7_0 --gpu 2 --config chatGPT_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0    >/dev/null 2>&1 &  
# nohup python run.py --ID USMLE_7_2 --gpu 0 --config chatGPT_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0   --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_7_3 --gpu 1 --config chatGPT_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  
# nohup python run.py --ID USMLE_7_5 --gpu 2 --config chatGPT_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1  --if_hierarchical_retrieval True  --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 


# nohup python run.py --ID MedMCQA_7_0 --gpu 3 --config chatGPT_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0    >/dev/null 2>&1 &  
# nohup python run.py --ID MedMCQA_7_2 --gpu 3 --config chatGPT_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0   --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 
# nohup python run.py --ID MedMCQA_7_3 --gpu 4 --config chatGPT_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  
# nohup python run.py --ID MedMCQA_7_5 --gpu 5 --config chatGPT_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1  --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 



# nohup python run.py --ID HEADQA_7_0 --gpu 4 --config chatGPT_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0    >/dev/null 2>&1 &  
# nohup python run.py --ID HEADQA_7_2 --gpu 6 --config chatGPT_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0   --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 
# nohup python run.py --ID HEADQA_7_3 --gpu 7 --config chatGPT_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &  
# nohup python run.py --ID HEADQA_7_5 --gpu 0 --config chatGPT_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1  --if_hierarchical_retrieval True --multi_query True  --rewrite_num 1 --infer_retri_num 5 --train_retri_num 5 >/dev/null 2>&1 & 

====================================================================================================================================================================================================

nohup python run.py --ID MedMCQA_7_0_test --gpu 3 --config llama2-7b_MedMCQA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0   >/dev/null 2>&1 &  
nohup python run.py --ID MedMCQA_7_1_test --gpu 0 --config llama2-7b_MedMCQA_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0   >/dev/null 2>&1 &  

====================================================================================================================================================================================================


# 7B
# nohup python run.py --ID USMLE_1  --gpu 0 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 1 --train_retri_num 1   --epoch 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_2  --gpu 1 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 2 --train_retri_num 2   --epoch 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_3  --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 3 --train_retri_num 3   --epoch 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_4  --gpu 3,4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 4 --train_retri_num 4   --epoch 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_6  --gpu 5,6,7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 6 --train_retri_num 6   --epoch 1 --test_batch_size 1 --train_batch_size 1  --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_7  --gpu 5,6,7 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 7 --train_retri_num 7   --epoch 1 --train_batch_size 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_8  --gpu XX --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 8 --train_retri_num 8   --epoch 1 --train_batch_size 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_9  --gpu XX --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 9 --train_retri_num 9   --epoch 1 --train_batch_size 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_10 --gpu XX --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 10 --train_retri_num 10 --epoch 1 --train_batch_size 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 


# 13B
# nohup python run.py --ID USMLE_01  --gpu 2,3 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 1 --train_retri_num 1   --epoch 1 --train_batch_size 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_02  --gpu 0,1 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 2 --train_retri_num 2   --epoch 1 --train_batch_size 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_03  --gpu 2,3 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 3 --train_retri_num 3   --epoch 1 --train_batch_size 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
nohup python run.py --ID USMLE_04  --gpu 4,5,6 --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 4 --train_retri_num 4   --epoch 1 --train_batch_size 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 

# nohup python run.py --ID USMLE_06  --gpu XX --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 6 --train_retri_num 6   --epoch 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_07  --gpu XX --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 7 --train_retri_num 7   --epoch 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_08  --gpu XX --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 8 --train_retri_num 8   --epoch 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_09  --gpu XX --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 9 --train_retri_num 9   --epoch 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 
# nohup python run.py --ID USMLE_10  --gpu XX --config llama2-13b_USMLE_MI_RA.yaml --dataset USMLE --infer_retri_num 10 --train_retri_num 10 --epoch 1 --test_batch_size 1 --if_hierarchical_retrieval True --multi_query True  >/dev/null 2>&1 & 



# 36.92
nohup python run.py --ID USMLE_RA_noise   --gpu 1 --config llama2-7b_USMLE_RA.yaml      --dataset USMLE --retrieval_corpus_ids 2  --infer_retri_num 4    >/dev/null 2>&1 & 
nohup python run.py --ID USMLE_RA_gold    --gpu 2,7 --config llama2-7b_USMLE_RA.yaml    --dataset USMLE --infer_add_gold_retrieval True >/dev/null --infer_retri_num 4 2>&1 & 
nohup python run.py --ID USMLE_ADRA_noise --gpu 3,4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --retrieval_corpus_ids 2        --epoch 1 --if_hierarchical_retrieval True --multi_query True  --infer_retri_num 4 --train_retri_num 4 >/dev/null 2>&1 & 
nohup python run.py --ID USMLE_ADRA_gold  --gpu 5,6 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --infer_add_gold_retrieval True --epoch 1 --if_hierarchical_retrieval True --multi_query True  --infer_retri_num 4 --train_retri_num 4 >/dev/null 2>&1 & 


nohup python run.py --ID MedMCQA_RA_noise   --gpu 1 --config llama2-7b_MedMCQA_RA.yaml    --dataset MedMCQA --retrieval_corpus_ids 2      >/dev/null 2>&1 & 
nohup python run.py --ID MedMCQA_RA_gold    --gpu 2 --config llama2-7b_MedMCQA_RA.yaml    --dataset MedMCQA --infer_add_gold_retrieval True --infer_retri_num 4 >/dev/null 2>&1 & 
nohup python run.py --ID MedMCQA_ADRA_noise --gpu X --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --retrieval_corpus_ids 2      --epoch 1 --if_hierarchical_retrieval True --multi_query True   --infer_retri_num 4 --train_retri_num 4   >/dev/null 2>&1 &
nohup python run.py --ID MedMCQA_ADRA_gold  --gpu X --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --infer_add_gold_retrieval True --epoch 1 --if_hierarchical_retrieval True --multi_query True --infer_retri_num 4 --train_retri_num 4   >/dev/null 2>&1 & 


nohup python run.py --ID HEADQA_RA_noise   --gpu X --config llama2-7b_HEADQA_RA.yaml    --dataset HEADQA --retrieval_corpus_ids 2  --infer_retri_num 4    >/dev/null 2>&1 & 
nohup python run.py --ID HEADQA_RA_gold    --gpu X --config llama2-7b_HEADQA_RA.yaml    --dataset HEADQA --infer_add_gold_retrieval True >/dev/null 2>&1 & 
nohup python run.py --ID HEADQA_ADRA_noise --gpu X --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --retrieval_corpus_ids 2      --epoch 1 --if_hierarchical_retrieval True --multi_query True   --infer_retri_num 4 --train_retri_num 4  >/dev/null 2>&1 & 
nohup python run.py --ID HEADQA_ADRA_gold  --gpu X --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --infer_add_gold_retrieval True --epoch 1 --if_hierarchical_retrieval True --multi_query True --infer_retri_num 4 --train_retri_num 4 >/dev/null 2>&1 & 






