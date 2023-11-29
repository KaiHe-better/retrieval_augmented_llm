nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE.yaml  >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_document_num 3 >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_document_num 5 >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_document_num 7 >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-7b_USMLE_RA.yaml --max_document_num 9 >/dev/null 2>&1 &


nohup python run.py --gpu 5,6,7 --config llama2-70b_USMLE.yaml  >/dev/null 2>&1 &
nohup python run.py --gpu 5,6,7 --config llama2-70b_USMLE_RA.yaml  >/dev/null 2>&1 &
                                    