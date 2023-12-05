import os
import yaml
import sys
import argparse



parser = argparse.ArgumentParser()

# system settings
parser.add_argument('--ID', type=str, default='0', help='run ID')
parser.add_argument("--config", type=str, default="llama2-7b_USMLE_RA.yaml", help="Path to the config file")
parser.add_argument('--gpu', default="5,6,7", type=str, help='gpu device numbers')
parser.add_argument('--seed', default=42, help='trandom seed')
parser.add_argument('--num_workers', default=16, type=int, help='data_loader_work')

# model and name 
parser.add_argument("--if_train", type=bool, default=True, help="if retrieval augmented")
parser.add_argument("--if_RA", type=bool, default=False, help="if retrieval augmented")
parser.add_argument("--int8", type=bool, default=False, help="if int8")
parser.add_argument("--LLM", type=str,  default="llama2-7b", choices=["llama2-7b", "X", ], help="LLM to use")
parser.add_argument("--triever", type=str,  default="dragon+", choices=["dragon+", "NIL", ], help="triever to use")

# data
parser.add_argument('--dataset', type=str, default="USMLE", choices=["USMLE", "X", ], help='train_file_path')
parser.add_argument('--prompt_file', type=str, default="prompts/USMLE.json",  help='prompt_file')
parser.add_argument('--retrieval_raw_data_dir', type=str, default="datasets/USMLE/textbooks/en", help='retrieval_raw_data_dir')
parser.add_argument('--retrieval_processed_file_dir', type=str, default="datasets/USMLE/process_retrieval_corpus/", help='retrieval_processed_file_dir')

# retrieval
parser.add_argument('--retri_batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--max_retri_num', type=int, default=1, help='max_document_num')
parser.add_argument('--chunk_size', type=int, default=512, help='chunk_sizen, not token length')
parser.add_argument('--chunk_overlap', type=int, default=20, help='chunk_size')
parser.add_argument('--similarity_threshold', type=float, default=0.7, help='similarity_threshold')
parser.add_argument('--multi_query', type=bool, default=False, help='multi_query, using open AI')

# train
parser.add_argument('--max_train_retri_num', type=int, default=1, help='max_document_num')
parser.add_argument('--train_batch_size', type=int, default=2, help='train_batch_size')
parser.add_argument('--test_batch_size', type=int, default=4, help='test_batch_size')
parser.add_argument('--demonstration', type=bool, default=False, help='in_context learning')
parser.add_argument('--demons_cnt', type=int, default=1, help='demonstration number')
parser.add_argument('--retrieval_tau', type=float, default=1, help='demonstration number')
parser.add_argument('--llm_tau', type=float, default=1, help='demonstration number')
parser.add_argument('--l2_coef', type=float, default=1e-5, help='l2')
parser.add_argument('--lr', type=float, default=1e-5, help='lr for retriever')
parser.add_argument('--train_eval', type=int, default=20, help='lr for retriever')
parser.add_argument('--epoch', type=int, default=99999, help='lr for retriever')

# Decoding
parser.add_argument("--temperature", type=float, default=1e-9, help="Temperature for decoding")
parser.add_argument("--top_p", type=float, default=0, help="Nucleus sampling top-p")
parser.add_argument("--max_new_tokens", type=int, default=1, help="Max number of new tokens to generate in one step")
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
from utils.utils import load_LLM, load_retriever, get_logger, make_log_dir, seed_everything
from models.my_model import My_Model

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
args.device = device

seed_everything(42)


dir_path = make_log_dir()
args.dir_path = dir_path
args.print_logger = get_logger(dir_path, "print")
args.result_logger = get_logger(dir_path, "result")

def custom_excepthook(exc_type, exc_value, exc_traceback):
    args.print_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = custom_excepthook

args.print_logger.info("**************** Configuration **************** ")
for k in args.__dict__:
    args.print_logger.info(f"{k}: {args.__dict__[k]}")
args.print_logger.info("**************** Configuration **************** \n\n")



def main(args):
    LLM, LLM_tokenizer, stop_token_ids = load_LLM(args)
    retri_encoder, triever_tokenizer = load_retriever(args, args.print_logger)
    my_model = My_Model(LLM, LLM_tokenizer,  args,  stop_token_ids)

    train_data_loader, dev_data_loader, test_data_loader = get_loader(args)
    
    trainer = My_Trainer(args, my_model, LLM, LLM_tokenizer, retri_encoder, triever_tokenizer, device)
    
    if args.if_train:
        trainer.train_proc(train_data_loader, dev_data_loader)
    trainer.test_proc(test_data_loader, dev_data_loader)
     
    



if __name__ == "__main__":
    main(args)
