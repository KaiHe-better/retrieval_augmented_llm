nohup python run.py --gpu 5,6,7 --config llama2-70b_USMLE_RA.yaml --max_retri_num 3 >/dev/null 2>&1 &


nohup python run.py --gpu 5,6,7 --config llama2-70b_USMLE_RA.yaml --max_retri_num 3  >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-13b_USMLE_RA.yaml >/dev/null 2>&1 &


nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA_NIL.yaml   >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA_NIL.yaml --demonstration true --demons_cnt 1 >/dev/null 2>&1 &