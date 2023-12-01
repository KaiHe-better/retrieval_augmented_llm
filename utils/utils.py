import logging
from datetime import datetime
import sys
import torch
import os
import time
import shutil
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List
from langchain.chat_models import ChatOpenAI


class IndexRefreshScheduler(object):
    def __init__(self, format_str: str, freeze_retriever_steps: int, train_retriever: bool, logger):
        """Build an index refresh scheduler

        format_str: string that specifies the schedule.
            has the format: startstep-endstep:refreshrate,startstep-endstep:refreshrate
            e.g. format_str="0-100:10,100-1000000:500" will refresh the index every 10 steps for the first 100 steps
            and then every 500 steps from step 100 to 1M.

            Syntactic Sugar for a fixed schedule: can just pass in a single number
            e.g. format_str="100" will refresh the index every 100 steps

            -1 to never refresh
        )
        """
        self.logger = logger
        self.format_str = format_str
        self.train_retriever = train_retriever
        self.freeze_retriever_steps = freeze_retriever_steps
        self.steps2rates = IndexRefreshScheduler.parse_index_refresh_schedule_string(format_str)

    @classmethod
    def parse_index_refresh_schedule_string(cls, format_str):
        parsed = []
        if format_str == "-1":
            parsed = [(0, 2**32, 2**32)]
        elif format_str.isdigit():
            parsed = [(0, 2**32, int(format_str))]
        else:
            for piece in format_str.split(","):
                startend, rate = piece.split(":")
                start, end = startend.split("-")
                parsed.append((int(start), int(end), int(rate)))
        return parsed

    def is_time_to_refresh(self, step):
        if not (self.train_retriever or step == 0):  # if retriever is not trained only refresh at step 0
            return False
        if not step == 0 and step < self.freeze_retriever_steps:  # freeze first steps
            return False
        for st, en, rate in self.steps2rates:
            if st <= step < en:
                steps_since_refresh_schedule_change = step - st
                return (steps_since_refresh_schedule_change % rate) == 0
        self.logger.warn(
            "cant calculate refresh rate for this step, I dont have data here"
            " its likely training step is higher than the specificed refresh rate see --index_refresh_rate for help."
        )
        return False

class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

def extracted_label(res):
    res = res[:10]
    if "A" in res:
        return 0
    if "B" in res:
        return 1
    if "C" in res:
        return 2
    if "D" in res:
        return 3
    return 0    

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_log_dir():
    sys_path = str(sys.argv[1:])[1:-1].replace("'", "").replace("--", "").replace(",", "_").replace(" ", "") if len(str(sys.argv[1:]))>2 else str(sys.argv[1:])
    dir_path ="./results/output/"+sys_path[:80].replace("/","_")

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

    return dir_path

def get_logger(dir, name):
    
    logger = logging.getLogger(name)

    if name =="result":
        # 创建一个handler，用于写入日志文件
        filename = f'{datetime.now().date()}_{name}.log'
        filename = os.path.join(dir, filename)
        fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')

    if name =="print":
        # 创建一个handler，用于写入日志文件
        filename = f'{datetime.now().date()}_{name}.log'
        filename = os.path.join(dir, filename)
        fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')

        # 再创建一个handler用于输出到控制台
        ch = logging.StreamHandler()

    # 定义输出格式(可以定义多个输出格式例formatter1，formatter2)
    # formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # 定义日志输出层级
    logger.setLevel(logging.DEBUG)
    
    if name =="result":
        # 定义控制台输出层级
        # logger.setLevel(logging.DEBUG)
        # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
        fh.setFormatter(formatter)
        # 给logger对象绑定文件操作符
        logger.addHandler(fh)

    if name =="print":
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch.setFormatter(formatter)

        # 给logger对象绑定文件操作符
        logger.addHandler(ch)

    return logger

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def load_LLM(args, dtype=torch.float16):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Llama: set up the root dir
    if args.LLM == "chatGPT":
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=args.temperature) 
        return model, "tokenizer", "stop_token_ids"
    else:
        model_name_or_path = args.LLM
        open_source_models = ["gpt2", "llama", "alpaca", "vicuna"]
        if any([m in model_name_or_path for m in open_source_models]):
            if model_name_or_path == "gpt2":
                model_name_or_path = "gpt2"

            if model_name_or_path == "llama2-7b":
                model_name_or_path = "../LLM_models/llama2/Llama-2-7b-chat-hf"

            if model_name_or_path == "llama2-13b":
                model_name_or_path = "../LLM_models/llama2/Llama-2-13b-chat-hf"

            if model_name_or_path == "llama2-70b":
                model_name_or_path = "../LLM_models/llama2/Llama-2-70b-chat-hf"

            model_name_or_path = os.path.join(model_name_or_path)
        else:
            raise Exception(" Please choose valid LLM ! ")



        # Load the FP16 model
        args.print_logger.info(f"Loading {model_name_or_path} in {dtype}...")
        if args.int8:
            args.print_logger.warn("Use LLM.int8")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            torch_dtype=dtype,
            max_memory=get_max_memory(),
            load_in_8bit=args.int8,
            offload_folder=model_name_or_path,
            # pretraining_tp=8
        )
        args.print_logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        tokenizer.padding_side = "left"
        
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.cls_token = tokenizer.bos_token
        tokenizer.mask_token = tokenizer.eos_token
        tokenizer.sep_token = tokenizer.eos_token
        
        stop = ["\n", "\n\n"]
        stop = list(set(stop + ["Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = list(set([tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [model.config.eos_token_id]))
        if "llama" in args.LLM:
            stop_token_ids.remove(tokenizer.unk_token_id)


        # Fix OPT bos token problem in HF
        if "opt" in model_name_or_path:
            tokenizer.bos_token = "<s>"
        
        # model.save_pretrained("/raid/hpc/hekai/WorkShop/My_project/LLM_models/llama2/Llama-2-70b-chat-hf")
        # tokenizer.save_pretrained("/raid/hpc/hekai/WorkShop/My_project/LLM_models/llama2/Llama-2-70b-chat-hf")
        
        return model, tokenizer, stop_token_ids

def load_retriever(args, print_logger):
    if not args.if_RA:
        return "", ""
    else:
        print_logger.info("loading retriever ...")
        if args.triever == "dragon+":
            tokenizer_path = "../LLM_models/dragon+/facebook_dragon-plus-query-encoder"
            query_encoder_path = "../LLM_models/dragon+/facebook_dragon-plus-query-encoder"
            context_encoder_path = "../LLM_models/dragon+/facebook_dragon-plus-context-encoder"
            retri_encoder_path = (query_encoder_path, context_encoder_path)

        if args.triever == "NIL":
            tokenizer_path = "../LLM_models/google/t5_xxl_true_nli_mixture"
            retri_encoder_path = "../LLM_models/google/t5_xxl_true_nli_mixture"


        triever_tokenizer =  AutoTokenizer.from_pretrained(tokenizer_path)
        query_encoder =  AutoModel.from_pretrained('facebook/dragon-plus-query-encoder')
        context_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder')

        return (query_encoder, context_encoder), triever_tokenizer
    

