import os
import yaml
import sys
import argparse



parser = argparse.ArgumentParser()

# system settings
parser.add_argument('--ID', default='0', help='run ID')
parser.add_argument("--config", type=str, default="llama2-7b_USMLE.yaml", help="Path to the config file")
parser.add_argument('--gpu', default="5,6,7", type=str, help='gpu device numbers')
parser.add_argument('--seed', default=42, help='trandom seed')
parser.add_argument('--num_workers', default=16, type=int, help='data_loader_work')

# model and name 
parser.add_argument("--if_RA", type=bool, default=False, help="if retrieval augmented")
parser.add_argument("--LLM", type=str,  default="llama2-7b", choices=["llama2-7b", "X", ], help="LLM to use")
parser.add_argument("--triever", type=str,  default="dragon+", choices=["dragon+", "X", ], help="triever to use")

# data
parser.add_argument('--dataset', type=str, default="USMLE", choices=["USMLE", "X", ], help='train_file_path')
parser.add_argument('--prompt_file', type=str, default="prompts/USMLE.json",  help='prompt_file')
parser.add_argument('--retrieval_raw_data_dir', type=str, default="datasets/USMLE/textbooks/en", help='retrieval_raw_data_dir')
parser.add_argument('--retrieval_processed_file_dir', type=str, default="datasets/USMLE/process_retrieval_corpus/", help='retrieval_processed_file_dir')
parser.add_argument('--chunk_size', type=int, default=512, help='chunk_sizen, not token length')
parser.add_argument('--chunk_overlap', type=int, default=50, help='chunk_size')

# retrieval
parser.add_argument('--similarity_threshold', type=float, default=0.7, help='similarity_threshold')
parser.add_argument('--multi_query', type=bool, default=False, help='multi_query, using open AI')

# train
parser.add_argument('--retri_batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--demonstration', type=bool, default=False, help='in_context learning')
parser.add_argument('--demons_cnt', type=int, default=1, help='demonstration number')

# Decoding
parser.add_argument("--temperature", type=float, default=0, help="Temperature for decoding")
parser.add_argument("--top_p", type=float, default=0, help="Nucleus sampling top-p")
parser.add_argument("--max_new_tokens", type=int, default=100, help="Max number of new tokens to generate in one step")
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
    retri_query_encoder_path,  retri_context_encoder_path, triever_tokenizer_path = load_retriever(args, args.print_logger)
    my_LLM, LLM_tokenizer, _ = load_LLM(args)
    
    train_data_loader, dev_data_loader, test_data_loader = get_loader(args)
    
    trainer = My_Trainer(args, my_LLM, LLM_tokenizer, retri_query_encoder_path,  retri_context_encoder_path, triever_tokenizer_path, device)
    trainer.train_proc(train_data_loader, dev_data_loader, test_data_loader)
     
    



if __name__ == "__main__":
    main(args)
    