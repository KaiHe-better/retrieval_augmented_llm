import os
import yaml
import sys
import argparse

parser = argparse.ArgumentParser()
 
# system settings
parser.add_argument('--ID', type=str, default='0', help='run ID')
parser.add_argument("--config", type=str, default="llama2-7b_USMLE_RA_test.yaml", help="Path to the config file")
parser.add_argument('--gpu', default="0", type=str, help='gpu device numbers')
parser.add_argument('--seed', default=42, help='trandom seed')
parser.add_argument('--num_workers', default=16, type=int, help='data_loader_work')
parser.add_argument("--test_code_flag", type=bool, default=False, help="if retrieval augmented")
parser.add_argument("--retriever_ckpt_path", type=str, default=None, help="if load trained retriever")

# model and name 
parser.add_argument("--if_train", type=bool, default=True, help="if retrieval augmented")
parser.add_argument("--if_RA", type=bool, default=True, help="if retrieval augmented")
parser.add_argument("--int8", type=bool, default=False, help="if int8")
parser.add_argument("--LLM", type=str,  default="llama2-7b", choices=["llama2-7b", "X", ], help="LLM to use")
parser.add_argument("--triever", type=str,  default="dragon+", choices=["dragon+", "NIL", ], help="triever to use")

# data
parser.add_argument('--dataset', type=str, default="USMLE", choices=["USMLE", "X"], help='train_file_path')
parser.add_argument('--prompt_file', type=str, default="prompts/USMLE.json",  help='prompt_file')
parser.add_argument('--retrieval_raw_data_dir', type=str, default="datasets/USMLE/textbooks/en", help='retrieval_raw_data_dir')
parser.add_argument('--retrieval_processed_file_dir', type=str, default="datasets/USMLE/process_retrieval_corpus/", help='retrieval_processed_file_dir')
parser.add_argument('--preprocess_retri_num', type=int, default=100, help='max_document_num')

# retrieval
parser.add_argument('--infer_retri_num', type=int, default=5, help='max_document_num')
parser.add_argument('--multi_query', type=bool, default=False, help='multi_query, using open AI')
parser.add_argument('--rewrite_num', type=int, default=3, help='max_document_num')
parser.add_argument('--retri_batch_size', type=int, default=640, help='batch_size')
parser.add_argument('--similarity_threshold', type=float, default=0.7, help='similarity_threshold')
parser.add_argument('--chunk_size', type=int, default=512, help='chunk_sizen, not token length')
parser.add_argument('--chunk_overlap', type=int, default=20, help='chunk_size')
parser.add_argument('--save_ratio', type=float, default=0.8, help='chunk_size')

# train
parser.add_argument('--train_retri_num', type=int, default=5, help='max_document_num')
parser.add_argument('--train_batch_size', type=int, default=2, help='train_batch_size')
parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation_steps')
parser.add_argument('--test_batch_size', type=int, default=2, help='test_batch_size')
parser.add_argument('--demonstration', type=bool, default=False, help='in_context learning')
parser.add_argument('--demons_cnt', type=int, default=1, help='demonstration number')
parser.add_argument('--retrieval_tau', type=float, default=1, help='demonstration number')
parser.add_argument('--llm_tau', type=float, default=1, help='demonstration number')
parser.add_argument('--l2_coef', type=float, default=1e-5, help='l2')
parser.add_argument('--lr', type=float, default=1e-4, help='lr for retriever')
parser.add_argument('--train_eval', type=int, default=300, help='lr for retriever')
parser.add_argument('--epoch', type=int, default=99999, help='lr for retriever')
parser.add_argument('--confirm_enhanced_acc', type=bool, default=True, help='confirm_enhanced_acc')

# model parameters
parser.add_argument('--d_model', type=int, default=768, help='MI_learner dim')
parser.add_argument('--dim_feedforward', type=int, default=2048, help='MI_learner linear dim')
parser.add_argument('--layer_norm_eps', type=float, default=1e-5, help='MI_learner dim')
parser.add_argument('--nhead', type=int, default=8, help='MI_learner nhead')
parser.add_argument('--dropout', type=float, default=0.1, help='MI_learner dropout')

# decoding
parser.add_argument("--temperature", type=float, default=1e-9, help="Temperature for decoding")
parser.add_argument("--top_p", type=float, default=0, help="Nucleus sampling top-p")
parser.add_argument("--max_new_tokens", type=int, default=2, help="Max number of new tokens to generate in one step")
parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")

args = parser.parse_args()
args.config = "configs/"+args.config
config = yaml.safe_load(open(args.config)) 
parser.set_defaults(**config)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OPENAI_API_KEY"] = "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"

from dataloader.data_loader import get_loader  
from trainer import My_Trainer 
import torch
from utils.utils import load_LLM, load_retriever, get_logger, make_log_dir, seed_everything, process_document
from models.my_model import My_MI_learner

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
args.device = device

seed_everything(42)


dir_path = make_log_dir()
args.dir_path = dir_path
args.print_logger = get_logger(dir_path, "print")
args.test_result_logger = get_logger(dir_path, "test_result")
args.train_result_logger = get_logger(dir_path, "train_result")

def custom_excepthook(exc_type, exc_value, exc_traceback):
    args.print_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = custom_excepthook

args.print_logger.info("**************** Configuration **************** ")
for k in args.__dict__:
    args.print_logger.info(f"{k}: {args.__dict__[k]}")
args.print_logger.info("**************** Configuration **************** \n\n")



def main(args):
    if args.if_RA:
        retri_encoder, triever_tokenizer = load_retriever(args)
        all_retrieve_doc, text_splitter = process_document(args, triever_tokenizer)
    else:
        retri_encoder, triever_tokenizer, all_retrieve_doc, text_splitter=None, None, None, None

    LLM, LLM_tokenizer, stop_token_ids = load_LLM(args)
    MI_learner = My_MI_learner(args)

    train_data_loader, dev_data_loader, test_data_loader = get_loader(args)
    
    trainer = My_Trainer(args, MI_learner, LLM, LLM_tokenizer, device, retri_encoder, triever_tokenizer, all_retrieve_doc, text_splitter)
    
    if args.if_train:
        trainer.train_proc(train_data_loader, dev_data_loader, test_data_loader)

    trainer.test_proc(test_data_loader, dev_data_loader)
    
    



if __name__ == "__main__":
    main(args)
