nohup python run.py --gpu 5,6,7 --config llama2-13b_USMLE_RA.yaml --similarity_threshold 0.65 >/dev/null 2>&1 &


nohup python run.py --gpu 5,6,7 --config llama2-13b_USMLE_RA.yaml --max_document_num 1 >/dev/null 2>&1 &
                                    