nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 10 --train_batch_size 16 >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 50 --train_batch_size 16 >/dev/null 2>&1 &



nohup python run.py --ID 000 --gpu 5 --config llama2-7b_USMLE_RA.yaml >/dev/null 2>&1 &

nohup python run.py --ID 0 --gpu 6 --config llama2-7b_USMLE_RA.yaml  --train_batch_size 4 --lr 1e-5 >/dev/null 2>&1 &
nohup python run.py --ID 1 --gpu 7 --config llama2-7b_USMLE_RA.yaml  --train_batch_size 4 --lr 1e-6 >/dev/null 2>&1 &



nohup python run.py --ID 00 --gpu 5 --config llama2-7b_USMLE_RA.yaml  --infer_retri_num 10 >/dev/null 2>&1 &

# nohup python run.py --ID 0 --gpu 5 --config llama2-7b_USMLE_RA.yaml --retriever_ckpt_path "./results/output/ID_0_gpu_5_config_llama2-7b_USMLE_RA.yaml_max_train_retri_num_5_train_batch_size/query_encoder.pkl" >/dev/null 2>&1 &
# nohup python run.py --ID 1 --gpu 6 --config llama2-7b_USMLE_RA.yaml --retriever_ckpt_path "./results/output/ID_1_gpu_6_config_llama2-7b_USMLE_RA.yaml_max_train_retri_num_20_train_batch_siz/query_encoder.pkl" >/dev/null 2>&1 &
# nohup python run.py --ID 2 --gpu 7 --config llama2-7b_USMLE_RA.yaml --retriever_ckpt_path "./results/output/ID_2_gpu_7_config_llama2-7b_USMLE_RA.yaml_max_train_retri_num_20_train_batch_siz/query_encoder.pkl" >/dev/null 2>&1 &

