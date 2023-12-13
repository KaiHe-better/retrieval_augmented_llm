nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 10 --train_batch_size 16 >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 50 --train_batch_size 16 >/dev/null 2>&1 &



nohup python run.py --ID 0 --gpu 5 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 5  --train_batch_size 8 >/dev/null 2>&1 &
nohup python run.py --ID 1 --gpu 6 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 20 --train_batch_size 8 >/dev/null 2>&1 &
nohup python run.py --ID 2 --gpu 7 --config llama2-7b_USMLE_RA.yaml --max_train_retri_num 20 --train_batch_size 1 >/dev/null 2>&1 &

