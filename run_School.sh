
nohup python run.py --ID 0 --gpu 0 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0   >/dev/null 2>&1 &
nohup python run.py --ID 1 --gpu 1 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1   >/dev/null 2>&1 &
nohup python run.py --ID 2 --gpu 2 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2   >/dev/null 2>&1 &
nohup python run.py --ID 4 --gpu 3 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1 --multi_query True  >/dev/null 2>&1 &
nohup python run.py --ID 4 --gpu 4 --config llama2-7b_USMLE_MI_RA.yaml --dataset USMLE --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  >/dev/null 2>&1 &

====================================================================================================================================================================================================

nohup python run.py --ID 0 --gpu 0 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0  >/dev/null 2>&1 &
nohup python run.py --ID 1 --gpu 1 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &
nohup python run.py --ID 2 --gpu 2 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 &
nohup python run.py --ID 3 --gpu 3 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1 --multi_query True  >/dev/null 2>&1 &
nohup python run.py --ID 4 --gpu 4 --config llama2-7b_HEADQA_MI_RA.yaml --dataset HEADQA --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  >/dev/null 2>&1 &

====================================================================================================================================================================================================

nohup python run.py --ID 0 --gpu 0 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0  >/dev/null 2>&1 &
nohup python run.py --ID 1 --gpu 1 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1  >/dev/null 2>&1 &
nohup python run.py --ID 2 --gpu 2 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1_2  >/dev/null 2>&1 &
nohup python run.py --ID 3 --gpu 3 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1 --multi_query True  >/dev/null 2>&1 &
nohup python run.py --ID 4 --gpu 4 --config llama2-7b_MedMCQA_MI_RA.yaml --dataset MedMCQA --epoch 1 --retrieval_corpus_ids 0_1_2  --multi_query True  >/dev/null 2>&1 &