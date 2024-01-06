# nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 10 --train_batch_size 16 >/dev/null 2>&1 &
# nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 50 --train_batch_size 16 >/dev/null 2>&1 &

# nohup python run.py --ID 0 --gpu 5 --config chatGPT_USMLE.yaml  >/dev/null 2>&1 &

# nohup python run.py --ID 0 --gpu 5 --config llama2-7b_USMLE_RA.yaml  --train_retri_num 9 --train_batch_size 4  >/dev/null 2>&1 &
# nohup python run.py --ID 0 --gpu 5 --config llama2-7b_USMLE_RA.yaml  --train_retri_num 10 --train_batch_size 8  >/dev/null 2>&1 &
# nohup python run.py --ID 0 --gpu 5 --config llama2-7b_USMLE_RA.yaml  --train_retri_num 5  --train_batch_size 8 >/dev/null 2>&1 &

# nohup python run.py --ID 1 --gpu 7 --config llama2-7b_USMLE_RA.yaml  --train_batch_size 4 --lr 1e-6 >/dev/null 2>&1 &

# nohup python run.py --ID 1 --gpu 6 --config chatGPT_USMLE_RA.yaml --multi_query True --chunk_size 256 --rewrite_num 2 --infer_retri_num 3 --test_batch_size 4 >/dev/null 2>&1 &
# nohup python run.py --ID 220 --gpu 6 --config chatGPT_USMLE_RA.yaml  --chunk_size 256  --infer_retri_num 6 --test_batch_size 4 >/dev/null 2>&1 &

# nohup python run.py --ID 1 --gpu 6 --config chatGPT_USMLE_RA.yaml --multi_query True --chunk_size 512 --rewrite_num 2 --infer_retri_num 3 --test_batch_size 4 >/dev/null 2>&1 &
# nohup python run.py --ID 221 --gpu 7 --config chatGPT_USMLE_RA.yaml  --chunk_size 512  --infer_retri_num 6 --test_batch_size 4 >/dev/null 2>&1 &

# nohup python run.py --ID 000 --gpu 6 --config llama2-7b_USMLE_RA.yaml  --chunk_size 256  --multi_query True --rewrite_num 2 --infer_retri_num 3 --test_batch_size 2 >/dev/null 2>&1 &
# nohup python run.py --ID 111 --gpu 7 --config llama2-7b_USMLE_RA.yaml  --chunk_size 512  --infer_retri_num 5 --test_batch_size 2 >/dev/null 2>&1 &

# nohup python run.py --ID 0 --gpu 5 --config llama2-7b_USMLE_RA.yaml --retriever_ckpt_path "./results/output/ID_0_gpu_5_config_llama2-7b_USMLE_RA.yaml_max_train_retri_num_5_train_batch_size/query_encoder.pkl" >/dev/null 2>&1 &
# nohup python run.py --ID 1 --gpu 6 --config llama2-7b_USMLE_RA.yaml --retriever_ckpt_path "./results/output/ID_1_gpu_6_config_llama2-7b_USMLE_RA.yaml_max_train_retri_num_20_train_batch_siz/query_encoder.pkl" >/dev/null 2>&1 &
# nohup python run.py --ID 2 --gpu 7 --config llama2-7b_USMLE_RA.yaml --retriever_ckpt_path "./results/output/ID_2_gpu_7_config_llama2-7b_USMLE_RA.yaml_max_train_retri_num_20_train_batch_siz/query_encoder.pkl" >/dev/null 2>&1 &

# nohup python run.py --ID 0 --gpu 5,6 --config llama2-70b_USMLE_RA_test.yaml --train_retri_num 7 --infer_retri_num 7 --lr 1e-4 --retri_batch_size 32 >/dev/null 2>&1 &

nohup python run.py --ID 0   --gpu 7 --config llama2-7b_USMLE_RA.yaml --train_batch_size 2 --test_batch_size 2 --train_retri_num 5 --infer_retri_num 5  >/dev/null 2>&1 &
nohup python run.py --ID 1   --gpu 2 --config llama2-7b_USMLE_RA.yaml --train_batch_size 2 --test_batch_size 2 --train_retri_num 5 --infer_retri_num 5  --accumulation_steps 2  >/dev/null 2>&1 &
nohup python run.py --ID 2   --gpu 3 --config llama2-7b_USMLE_RA.yaml --train_batch_size 2 --test_batch_size 2 --train_retri_num 5 --infer_retri_num 5  --accumulation_steps 3  >/dev/null 2>&1 &
nohup python run.py --ID 3   --gpu 4 --config llama2-7b_USMLE_RA.yaml --train_batch_size 2 --test_batch_size 2 --train_retri_num 5 --infer_retri_num 5  --accumulation_steps 4  >/dev/null 2>&1 &


nohup python run.py --ID 1_multihead --gpu 6 --config chatGPT_USMLE_RA_test.yaml --train_batch_size 3 --train_retri_num 7 --infer_retri_num 7 --lr 1e-4  >/dev/null 2>&1 &


