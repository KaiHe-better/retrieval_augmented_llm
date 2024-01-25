import os
import yaml
import sys
import argparse

parser = argparse.ArgumentParser()

# system settings
parser.add_argument('--ID', type=str, default='0', help='run ID')
parser.add_argument('--gpu', default="3", type=str, help='gpu device numbers')
parser.add_argument('--seed', default=42, help='trandom seed')
parser.add_argument('--num_workers', default=16, type=int, help='data_loader_work')
parser.add_argument("--test_code_flag", type=bool, default=False, help="if retrieval augmented")
parser.add_argument("--loading_ckpt_path", type=str, default=None, help="loading_ckpt_path, None ")
# model and name
parser.add_argument("--if_train", type=bool, default=True, help="if retrieval augmented")
parser.add_argument("--if_RA", type=bool, default=True, help="if retrieval augmented")
parser.add_argument("--if_MI_RA", type=bool, default=True, help="if_MI_RA")
parser.add_argument("--LLM", type=str,  default="llama2-7b", choices=["llama2-7b", "chatGPT", ], help="LLM to use")
parser.add_argument("--triever", type=str,  default="dragon+", choices=["dragon+", "NIL", ], help="triever to use")
parser.add_argument("--num_layers", type=int,  default=1, help="num_layers")
# data
parser.add_argument('--dataset', type=str, default="USMLE", choices=["USMLE", "MedMCQA", "HEADQA", "MMLU", "OTTQA"], help='train_file_path')
parser.add_argument("--config", type=str, default="llama2-7b_USMLE_MI_RA.yaml", help="Path to the config file")
parser.add_argument('--chunk_size', type=int, default=512, help='chunk_sizen, not token length')
parser.add_argument('--chunk_overlap', type=int, default=20, help='chunk_size')
# retrieval
parser.add_argument('--train_add_gold_retrieval', type=bool, default=False, help='max_document_num')
parser.add_argument('--infer_add_gold_retrieval', type=bool, default=False, help='max_document_num')
parser.add_argument('--infer_retri_num', type=int, default=5, help='max_document_num')
parser.add_argument('--pass_retri_num', type=int, default=3, help='max_document_num')
parser.add_argument('--test_batch_size', type=int, default=2, help='test_batch_size')
parser.add_argument('--multi_query', type=bool, default=False, help='multi_query, using open AI')
parser.add_argument('--rewrite_num', type=int, default=2, help='rewrite_num, more than 1')
parser.add_argument('--OTTQA_more_passage', type=bool, default=True, help='OTTQA_more_passage')
parser.add_argument('--retri_batch_size', type=int, default=320, help='batch_size')
parser.add_argument('--retrieval_corpus_ids', type=str, default="0", help='0, 0_1, 0_1_2')
# hierarchical retrieval
parser.add_argument('--if_hierarchical_retrieval', type=bool, default=True, help='if_hierarchical_retrieval')
parser.add_argument('--hierarchical_ratio', type=float, default=1.4, help='hierarchical_ratio')
parser.add_argument('--quantile_num', type=float, default=0.95, help='quantile_num')
# train
parser.add_argument('--train_retri_num', type=int, default=5, help='max_document_num')
parser.add_argument('--train_batch_size', type=int, default=2, help='train_batch_size')
parser.add_argument('--accumulation_steps', type=int, default=1, help='accumulation_steps')
parser.add_argument('--demonstration', type=bool, default=False, help='in_context learning')
parser.add_argument('--demons_cnt', type=int, default=1, help='demonstration number')
parser.add_argument('--l2_coef', type=float, default=0, help='l2')
parser.add_argument('--train_eval', type=int, default=100, help='lr for retriever')
parser.add_argument('--epoch', type=int, default=99999, help='lr for retriever')
# lr
parser.add_argument('--lr', type=float, default=1e-4, help='lr for retriever')
parser.add_argument('--init_lr_num', type=int, default=500, help='lr for retriever')
parser.add_argument('--lr_decay', type=float, default=0.9, help='lr for retriever')
parser.add_argument('--lr_decay_interval', type=int, default=400, help='lr for retriever')
# loss
parser.add_argument('--loss_list', type=str, default="kl_soft+kl_hard", help='kl_soft+kl_hard+mse')
parser.add_argument('--mse_weight', type=float, default=0, help='soft_weight')
parser.add_argument('--soft_weight', type=float, default=0.7, help='soft_weight')
parser.add_argument('--hard_weight', type=float, default=0.3, help='hard_weight')
# model parameters
parser.add_argument('--d_model', type=int, default=768, help='MI_learner dim')
parser.add_argument('--dim_feedforward', type=int, default=2048, help='MI_learner linear dim')
parser.add_argument('--layer_norm_eps', type=float, default=1e-5, help='MI_learner dim')
parser.add_argument('--nhead', type=int, default=8, help='MI_learner nhead')
parser.add_argument('--dropout', type=float, default=0.1, help='MI_learner dropout')
# decoding
parser.add_argument("--temperature", type=float, default=1e-9, help="Temperature for decoding")
parser.add_argument("--top_p", type=float, default=0, help="Nucleus sampling top-p")
# parser.add_argument("--length_penalty", type=float, default=0.01, help="length_penalty")
# parser.add_argument("--num_beams", type=float, default=10, help="num_beams")
parser.add_argument("--max_new_tokens", type=int, default=1, help="Max number of new tokens to generate in one step")
parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")

args = parser.parse_args()
args.config = "configs/"+args.dataset+ "/"+args.config
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
from models.my_model import My_MI_learner

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
args.device = device

seed_everything(int(args.seed))

args.prompt_file = "prompts/" + args.dataset + ".json"
assert args.rewrite_num>1
if args.if_train is True and args.if_MI_RA is False:
    raise Exception("if if_train is true, if_MI_RA have to be true ! ")


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
    if args.if_RA or args.if_MI_RA:
        retri_encoder, triever_tokenizer = load_retriever(args)
    else:
        retri_encoder, triever_tokenizer = None, None

    LLM, LLM_tokenizer = load_LLM(args)
    train_data_loader, dev_data_loader, test_data_loader = get_loader(args, LLM_tokenizer)
    
    MI_learner = My_MI_learner(args, LLM_tokenizer.vocab_size if args.LLM != "chatGPT" else 32000)
    
    if args.loading_ckpt_path is not None:
        args.print_logger.info(f"loading ckpt from {args.loading_ckpt_path} ! \n=================\n")
        MI_learner.load_state_dict(torch.load(args.loading_ckpt_path))

    trainer = My_Trainer(args, MI_learner, LLM, LLM_tokenizer, device, retri_encoder, triever_tokenizer)
    
    if args.if_train and args.if_RA and args.if_MI_RA and (args.dataset!= "MMLU"):
        trainer.train_proc(train_data_loader, dev_data_loader, test_data_loader)

    # test_data_loader = dev_data_loader
    trainer.test_proc(test_data_loader, dev_data_loader)
    
    



if __name__ == "__main__":
    main(args)
