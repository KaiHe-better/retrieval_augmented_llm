from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from transformers import pipeline
from langchain.chains import LLMChain
import torch.nn.functional as F
import json
import os
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import My_Metrics
from torch.utils.tensorboard import SummaryWriter
from utils.utils import extracted_token_id_label, LineListOutputParser, empty_logger_file, combine_doc, __dist__, left_pad_loss_logit


class My_Trainer:

    def __init__(self, args, MI_learner, LLM, LLM_tokenizer, device, retri_encoder=None, triever_tokenizer=None):
        self.args = args
        self.print_logger = args.print_logger
        self.test_result_logger = args.test_result_logger
        self.train_result_logger = args.train_result_logger

        self.device = device
        self.MI_learner = MI_learner

        self.LLM = LLM
        self.LLM_tokenizer = LLM_tokenizer
        self.my_metrics = My_Metrics()
        self.writer = SummaryWriter(args.dir_path+"/runs/")

        if self.args.if_train:
            self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            # self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        if self.args.if_RA or args.if_MI_RA:
            self.triever_tokenizer =  triever_tokenizer

            if args.triever in ["dragon+"]:
                self.retriever =  retri_encoder.to(self.device)
            else:
                raise Exception("wrong !")

            if self.args.if_train:
                param_list_MI_learner = list(self.MI_learner.parameters())
                self.optimizer = torch.optim.Adam( param_list_MI_learner, lr=self.args.lr, weight_decay=self.args.l2_coef)
                
                lr_lambda = lambda step: 1 if step < self.args.init_lr_num else self.args.lr_decay ** ((step - self.args.init_lr_num) // self.args.lr_decay_interval)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

            if self.args.demonstration:
                prompt_format = "retrieve-demonstration-prompt"
            else:
                prompt_format = "retrieve-prompt"
        else:
            if self.args.demonstration:
                prompt_format = "general-demonstration-prompt"
            else:
                prompt_format = "general-prompt"
        self.print_logger.info(f"prompt_format: {prompt_format} \n")    
            
        with open(self.args.prompt_file, "r") as f:
            prompt_text = json.load(f)[prompt_format]

        self.prompt = PromptTemplate.from_template(prompt_text)
        if args.LLM != "chatGPT":
            self.pipe = pipeline(
                    "text-generation",
                    model=LLM,
                    tokenizer=self.LLM_tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    device_map="auto",
                    output_scores=True, 
                    return_dict_in_generate=True ) 
        else:
            self.pipe = LLMChain(prompt=self.prompt, llm=LLM)

    def random_select_demonstration(self, data_loader, batch_size):
        demon_prompt_list = []
        for i in range(batch_size):
            demon_prompt = ""
            for index, item in enumerate(data_loader):
                if self.args.demons_cnt < index+1:
                    break
                demon_prompt += "Demonstration " + str(index)+"\n Question: {} \n Options: {} \n Answer: <{}> \n\n".format(item["question"][0], item["options"][0], item["answer"][0])
            demon_prompt_list.append(demon_prompt)
        return demon_prompt_list

    def write_chunked_common(self, process_file, text_splitter):
        all_doc = []
        retrieval_raw_data_dir =  "datasets/Retrieval_corpus/raw_retrieval_corpus/" 

        if self.args.retrieval_corpus_ids == "0":
            tmp_file_list = ["raw_retrieval_corpus_0"]
        elif self.args.retrieval_corpus_ids == "0_1":
            tmp_file_list = ["raw_retrieval_corpus_0", "raw_retrieval_corpus_1"]
        elif self.args.retrieval_corpus_ids == "0_1_2":
            tmp_file_list = ["raw_retrieval_corpus_0", "raw_retrieval_corpus_1", "raw_retrieval_corpus_2"]
        else:
            raise Exception("wrong corpus")
        
        for tmp_file_list_item in tmp_file_list:
            cur_path = os.path.join(retrieval_raw_data_dir, tmp_file_list_item)
            for foldername, subfolders, filenames in os.walk(cur_path):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    self.args.print_logger.info(f"chunking retrieval files: {filename} \n")
                    loader = TextLoader(file_path)
                    documents = loader.load()
                    chunks = text_splitter.split_documents(documents)
                    temp_list = [[re.sub(r'[^\x00-\x7F]+', ' ', i.page_content.replace("\n\n", "\n "))] for i in chunks]
                    all_doc+=temp_list

        with open(process_file, "w", encoding="utf-8") as f:
            for i in all_doc:
                f.writelines(str(i)+"\n")

    def write_chunked_OTTQA(self, process_file, text_splitter):
        self.args.print_logger.info(f"chunking retrieval files: {process_file} \n")
        raw_file = self.args.retrieval_raw_data_dir+ "_".join(process_file.split("/")[-1].split("_")[0:2]) +".json"
        loader = TextLoader(raw_file)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        temp_list = [[re.sub(r'[^\x00-\x7F]+', ' ', i.page_content.replace("\n\n", "\n "))] for i in chunks]

        with open(process_file, "w", encoding="utf-8") as f:
            for i in temp_list:
                f.writelines(str(i)+"\n")

    def process_document(self, process_file):
        start_time = time.time()
        
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.triever_tokenizer, chunk_size=self.args.chunk_size, chunk_overlap=self.args.chunk_overlap)
            
        if not os.path.exists(process_file):
            if self.args.dataset == "OTTQA":
                self.write_chunked_OTTQA(process_file, text_splitter)
            else: 
                self.write_chunked_common(process_file, text_splitter)
        else:
            self.args.print_logger.info(f"{process_file} already chunked !")
        
        with open(process_file, "r", encoding="utf-8") as f:
            all_doc = f.readlines()
        
        all_doc = [eval(i)[0] for i in all_doc]

        self.args.print_logger.info(f"loading retrieval files in %.2f mins. \n"% ((time.time() - start_time)/60))
        return all_doc
 
    def updata_retri_embedding(self):
        start_time = time.time()
        self.print_logger.info("updata_retri_embedding ...")
           
        if self.args.dataset == "OTTQA":
            file_types = ["train_passage", "train_table", "dev_passage", "dev_table", "test_passage", "test_table"]
        else:
            file_types = ["train"]

        retrieval_processed_file_dir = "datasets/Retrieval_corpus/process_retrieval_corpus/" 

        process_files = []
        process_embed_files = []
        
        
        if self.args.test_code_flag:
            self.args.print_logger.info("\n====test==============\n====test==============\n====test==============\n")
            for file_type in file_types:
                temp_file_path = "toy_"+file_type+"_"+str(self.args.triever)
                temp_file_path = temp_file_path + "_"+str(self.args.chunk_size)+"_"+str(self.args.chunk_overlap)+"_"
                temp_file_path = temp_file_path + str(self.args.retrieval_corpus_ids)+".txt"
                process_file = os.path.join(retrieval_processed_file_dir, temp_file_path)

                process_files.append(process_file)
                process_embed_files.append(process_file[:-3].replace("process_retrieval_corpus", "process_retrieval_embed") +"pt")
        else:
            for file_type in file_types:
                temp_file_path = file_type+"_"+str(self.args.triever)
                temp_file_path = temp_file_path + "_"+str(self.args.chunk_size)+"_"+str(self.args.chunk_overlap)+"_"
                temp_file_path = temp_file_path + str(self.args.retrieval_corpus_ids)+".txt"
                process_file = os.path.join(retrieval_processed_file_dir, temp_file_path)
            
                process_files.append(process_file)
                process_embed_files.append(process_file[:-3].replace("process_retrieval_corpus", "process_retrieval_embed") +"pt")

        self.vectordb = {}
        self.retrieved_document = {}
        for process_file, process_embed_file, file_type in zip(process_files, process_embed_files, file_types):
            retrieved_document = self.process_document(process_file)

            if os.path.exists(process_embed_file):
                vectordb = torch.load(process_embed_file).to(self.args.device)
                self.print_logger.info(f"exist retri_embedding for {file_type} , loading it in %.2f min. "% ((time.time() - start_time)/60))
            else:
                self.print_logger.info(f"encoding retri embedding for {file_type} ... \n" )
                with torch.no_grad():
                    context_batches = [retrieved_document[i:i + self.args.retri_batch_size] for i in range(0, len(retrieved_document), self.args.retri_batch_size)]
                    tmp_vectordb = []
                    for context in tqdm(context_batches):
                        ctx_input = self.triever_tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=self.args.chunk_size).to(self.device)
                        # tmp_res = torch.mean(self.retriever(**ctx_input).last_hidden_state * ctx_input["attention_mask"][:, :, None], dim=1)
                        tmp_res = self.retriever(**ctx_input).last_hidden_state[:, 0, :]
                        tmp_vectordb.append(tmp_res.detach().cpu().to(self.device))
                        torch.cuda.empty_cache()
                vectordb  = torch.cat((tmp_vectordb), dim=0).transpose(0, 1)
                torch.save(vectordb, process_embed_file)
                self.print_logger.info(f"no saved retri_embedding for {file_type}, make it in %.2f min. \n"% ((time.time() - start_time)/60))
            
            self.vectordb[file_type] = vectordb
            self.retrieved_document[file_type] = retrieved_document
            self.print_logger.info(f"vectordb {file_type}, size: {self.vectordb[file_type].size()}")
        
    def retrieve(self, query, retri_num, train_flag):
        if train_flag and retri_num<=1:
            raise Exception("train need retrieve more than one docs !")

        if self.args.multi_query:
            output_parser = LineListOutputParser()
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate  {rewrite_num} 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search. 
                Provide these alternative questions separated by newlines.
                Original question: {question}""" )

            llm_chain = LLMChain(llm= ChatOpenAI(temperature=0), prompt=QUERY_PROMPT, output_parser=output_parser)
            batch_queries = []
            for q in query:
                batch_queries.append(q)
                batch_queries += llm_chain({"question":q, "rewrite_num":self.args.rewrite_num})["text"] 

        else:
            batch_queries=query

        query_input = self.triever_tokenizer(batch_queries, truncation=True, return_tensors='pt', max_length=self.args.chunk_size, padding=True).to(self.device)
        query_embs = self.retriever(**query_input).last_hidden_state
        query_emb = query_embs[:, 0, :]

        if retri_num<1:
            return [], [], query_embs, query_input["attention_mask"]
            
        if self.args.dataset == "OTTQA":
            if train_flag==True:
                curr_vectordb = self.vectordb["train_table"]
                curr_retrieved_document = self.retrieved_document["train_table"]
                if self.args.OTTQA_more_passage:
                    curr_vectordb = torch.cat((curr_vectordb,  self.vectordb["train_passage"]), dim=-1)
                    curr_retrieved_document = curr_retrieved_document + self.retrieved_document["train_passage"]
            else:
                curr_vectordb = self.vectordb["dev_table"] 
                curr_retrieved_document = self.retrieved_document["dev_table"]
                if self.args.OTTQA_more_passage:
                    curr_vectordb = torch.cat((curr_vectordb,  self.vectordb["dev_passage"]), dim=-1)
                    curr_retrieved_document = curr_retrieved_document + self.retrieved_document["dev_passage"]
        else:
            curr_vectordb = self.vectordb["train"] 
            curr_retrieved_document = self.retrieved_document["train"]
        
        scores = __dist__(query_emb, curr_vectordb, method='dot')
        batch_select_index = torch.argsort(scores, descending=True)[:, :retri_num].tolist()

        if self.args.multi_query:
            # results from multi query are all needed
            merged_list = []
            i = 0
            while i < len(batch_select_index):
                # Merge n consecutive sublists into one sublist
                merged_sublist = []
                for sublist in batch_select_index[i:i + self.args.rewrite_num+1]:
                    merged_sublist.extend(sublist)
                
                merged_list.append(merged_sublist)
                i = self.args.rewrite_num+i+1
            batch_select_index = merged_list

        batch_infer_doc = []
        batch_item_list = []
        for docs in batch_select_index:
            retrieve_doc = [curr_retrieved_document[i] for i in docs]
            batch_item_list.append(retrieve_doc)
            tmp_str = combine_doc(retrieve_doc)        
            batch_infer_doc.append(tmp_str)

        return batch_infer_doc, batch_item_list, query_embs, query_input["attention_mask"]

    def return_input_dict(self, dev_data_loader, data_item, retrieve_docs):
        if self.args.dataset == "OTTQA":
            if self.args.if_RA:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': data_item["question"], "context": retrieve_docs, "demonstration": demonstration}
                else:
                    input_dict = {'question': data_item["question"], "context": retrieve_docs}
            else:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': data_item["question"], "demonstration": demonstration}
                else:
                    input_dict = {'question': data_item["question"], }
        else:
            if self.args.if_RA:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': data_item["question"], 'options': data_item["options"], "context": retrieve_docs, "demonstration": demonstration}
                else:
                    input_dict = {'question': data_item["question"], 'options': data_item["options"], "context": retrieve_docs}
            else:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': data_item["question"], 'options': data_item["options"],  "demonstration": demonstration}
                else:
                    input_dict = {'question': data_item["question"], 'options': data_item["options"]}
        return input_dict
    
    def add_gold_retrieval(self, retrieve_docs, data_item):
        len_retrieve_docs = len(retrieve_docs)
        for index, (question_item, options_item, answer_item) in enumerate(zip(data_item['question'], data_item['options'], data_item['answer'])):
            for item in options_item.split('. '):
                if item.strip().startswith('<' + answer_item + '>'):
                    answer_str = item.split('>')[1].strip()
                    temp_doc = "document (0): " + question_item + " The answer is " + answer_str + "\n\n"
            
            if len_retrieve_docs>=1:
                retrieve_docs[index] =  temp_doc + retrieve_docs[index] 
            else:
                retrieve_docs.append(temp_doc) 

        return retrieve_docs
    
    def train_proc(self, train_data_loader, dev_data_loader, test_data_loader):
        if (not self.args.if_RA) or (not self.args.if_MI_RA):
            raise Exception("need retrieve ! ")

        self.updata_retri_embedding()
        self.print_logger.info("Start training ... \n ")
        
        total_batch = len(train_data_loader)
        step_num = -1
        best_step = 0
        best_performce = 0
        eval_num = 0
        for epoch_num in range(self.args.epoch):
            for data_item in train_data_loader:
                self.MI_learner.train()
                step_num+=1
                question = data_item['question']
                labels = data_item['label']
                one_hot_labels = data_item['one_hot_label']
                batch_answer = data_item["answer"]

                retrieve_docs, bags_list, query_emb, att_mask = self.retrieve(question, self.args.train_retri_num, train_flag=True)
                if self.args.train_add_gold_retrieval:
                    retrieve_docs = self.add_gold_retrieval(retrieve_docs, data_item)

                input_dict = self.return_input_dict(dev_data_loader, data_item, retrieve_docs)
                with torch.no_grad():
                    _, _,  _, save_doc_num, batch_loss, batch_logit_log_softmax = self.pipeline_inference(input_dict, labels, batch_answer, training_flag=True, record_flag=False)

                for doc_index, doc_num in enumerate(save_doc_num):
                    bags_list[doc_index] = bags_list[doc_index][:doc_num]

                loss_list, new_retrieve_docs, select_doc_num = self.MI_learner(query_emb, att_mask, bags_list, batch_logit_log_softmax, one_hot_labels, batch_loss, self.retriever, self.triever_tokenizer, True)
                total_loss = loss_list[-1]

                total_loss.backward()
                # new
                old_doc_len =  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
                new_doc_len =  sum([len(i) for i in self.LLM_tokenizer(new_retrieve_docs)["input_ids"]]) / len(retrieve_docs)

                self.writer.add_scalar('Loss/total_loss', round(float(total_loss), 4), step_num)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], step_num)

                self.print_logger.info(f"epoch_num: {epoch_num}, training process num: {step_num}/{total_batch}, mse_loss: {round(float(loss_list[0]), 4)}, kl_soft_loss: {round(float(loss_list[1]), 4)}, kl_hard_loss: {round(float(loss_list[2]), 4)}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len}, best_step:{best_step}, best_performce: {best_performce}")
                                       
                if (step_num + 1) % self.args.accumulation_steps == 0:
                    self.optimizer.step()
                    # if self.optimizer.param_groups[0]['lr'] >= 1e-5:
                        # self.scheduler.step()
                    self.optimizer.zero_grad()

                if (step_num % self.args.train_eval==0) and step_num>1:
                # if (step_num % self.args.train_eval==0) :
                    eval_num +=1
                    self.train_result_logger = empty_logger_file(self.train_result_logger)

                    break_cnt = 2 if self.args.test_code_flag else None
                    with torch.no_grad():
                        self.MI_learner.eval()
                        test_performce = self.test_proc(test_data_loader, dev_data_loader, step_num, break_cnt=break_cnt)
                        self.MI_learner.train()

                    if test_performce>best_performce:
                        best_performce = test_performce
                        best_step = step_num

                        if step_num>10:
                            torch.save(self.MI_learner.state_dict(), self.args.dir_path+'/MI_' +str(best_performce)+'.pkl') 

                # if step_num % 1 ==0 :
                #     break
                        
            # if step_num ==100:
            #   break

    def test_proc(self, test_data_loader, dev_data_loader, eval_num=0, break_cnt=None):
        
        if ((self.args.if_RA or self.args.if_MI_RA) and (self.args.if_train is False)) or ( self.args.dataset =="MMLU"):
            self.updata_retri_embedding()
            
        self.print_logger.info("\n Start test ...  ")

        all_test_labels = []
        all_test_prediction_ids = []
        all_test_predictions = []
        all_test_answers = []
        old_doc_len = 0
        new_doc_len = 0
        total_hallucination_cnt = 0
        for index, data_item in enumerate(test_data_loader):
            if index%200==0:
                self.print_logger.info(f"testing process num: {index}")
            question = data_item['question']
            batch_label = data_item["label"]
            batch_answer = data_item["answer"]

            if self.args.if_RA:
                retrieve_docs, bags_list, query_emb, att_mask = self.retrieve(question, self.args.infer_retri_num, train_flag=False)
                if self.args.infer_add_gold_retrieval:
                    retrieve_docs = self.add_gold_retrieval(retrieve_docs, data_item)

                old_doc_len +=  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
                if self.args.if_MI_RA:
                    _, retrieve_docs, _ = self.MI_learner(query_emb, att_mask, bags_list, "batch_logit_log_softmax", "one_hot_labels", "batch_loss", self.retriever, self.triever_tokenizer, False)
                    new_doc_len +=  sum([len(i) for i in self.LLM_tokenizer(retrieve_docs)["input_ids"]]) / len(retrieve_docs)
            else:
                retrieve_docs = ""

            input_dict = self.return_input_dict(dev_data_loader, data_item, retrieve_docs)
            batch_pred, batch_id_pred, batch_hallucination_cnt, _, _, _ = self.pipeline_inference(input_dict, batch_label, batch_answer, training_flag=False, record_flag=True)
            total_hallucination_cnt+=batch_hallucination_cnt

            all_test_labels+=batch_label
            all_test_prediction_ids+=batch_id_pred

            all_test_predictions+=batch_pred
            all_test_answers+=batch_answer

            break_cnt = 2 if self.args.test_code_flag else None
            if break_cnt is not None and break_cnt<index:
                break
        
        old_doc_len = old_doc_len / len(test_data_loader)   
        new_doc_len = new_doc_len / len(test_data_loader)   

        if self.args.dataset == "OTTQA":
            test_f1 , test_EM =  self.my_metrics.get_raw_scores(all_test_predictions, all_test_answers)   
            
            self.args.print_logger.info(f"test: f1 {test_f1}, test: EM {test_EM}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len} \n ")
            record_performance = test_f1

            self.writer.add_scalar('Performance/test/EM', test_EM, eval_num )
            self.writer.add_scalar('Performance/test/f1', test_f1, eval_num )
        else:
            test_acc, test_precision, test_recall, test_f1 = self.my_metrics.acc_PRF(all_test_labels, all_test_prediction_ids)
            self.args.print_logger.info(f"test: acc {test_acc}, f1 {test_f1}, precision {test_precision}, recall {test_recall}, old_doc_len:{old_doc_len}, new_doc_len:{new_doc_len}, hallucination: {total_hallucination_cnt/len(test_data_loader)/len(question)} \n ")
            record_performance = test_acc

            self.writer.add_scalar('Performance/test/acc', test_acc, eval_num )
            self.writer.add_scalar('Performance/test/precision', test_precision, eval_num )
            self.writer.add_scalar('Performance/test/recall', test_recall, eval_num )
            self.writer.add_scalar('Performance/test/f1', test_f1, eval_num )

        return record_performance
    
    def pipeline_inference(self, input_dict, label, batch_answer, training_flag=False, record_flag=True):
        if self.args.LLM == "chatGPT":
            batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num = self.non_local_llm_infer(input_dict, label, batch_answer, training_flag, record_flag)
            batch_loss, batch_logit_log_softmax = 0, 0
        else:
            batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss, batch_logit_log_softmax= self.local_llm_infer(input_dict, label, batch_answer, training_flag, record_flag)

        return batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss, batch_logit_log_softmax

    def non_local_llm_infer(self, input_dict, label, batch_answer, training_flag=False, record_flag=True):
        batch_pred = []
        batch_id_pred = []
        keys = input_dict.keys()
        batch_hallucination_cnt = 0
        
        if training_flag:
            save_doc_num = [self.args.train_retri_num]*len(label)
        else:
            save_doc_num = [self.args.infer_retri_num]*len(label)

        for index2, values in enumerate(zip(*input_dict.values())):
            current_inputs = dict(zip(keys, values))
            try:
                pred = self.pipe(current_inputs)
            except:
                current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:-1])
                self.print_logger.info("too long context, we short one retrieval results !")
                save_doc_num[index2] = save_doc_num[index2]-1
                try:
                    pred = self.pipe(current_inputs)
                except:
                    current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:-1])
                    self.print_logger.info("too long context, we short two retrieval results !")
                    save_doc_num[index2] = save_doc_num[index2]-1
                    try:
                        pred = self.pipe(current_inputs)
                    except:
                        current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:5])
                        self.print_logger.info("too long context for many times, we only take first 5 retrieval results !")
                        pred =  self.pipe(current_inputs)
                        save_doc_num[index2] = 5

            pred = pred["text"]
            pred, id_pred, hallucination_cnt = self.pasrse_record_res(self.prompt.format(**current_inputs) , label[index2], pred, batch_answer[index2], training_flag, record_flag) 
            batch_pred.append(pred)  
            batch_id_pred.append(id_pred)
            batch_hallucination_cnt+=hallucination_cnt

        return batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num
    
    def local_llm_infer(self, input_dict, label, batch_answer, training_flag=False, record_flag=True):
        if training_flag:
            save_doc_num = [self.args.train_retri_num]*len(label)
        else:
            save_doc_num = [self.args.infer_retri_num]*len(label)
    
        my_input_list = []
        keys = input_dict.keys()
        for values in zip(*input_dict.values()):
            current_inputs = dict(zip(keys, values))
            my_input = self.prompt.format(**current_inputs)
            my_input_list.append(my_input)
        
        batch_id_pred = []
        batch_pred = []
        batch_hallucination_cnt = 0
        inputs = self.LLM_tokenizer(my_input_list, return_tensors="pt", padding=True).to(self.args.device)
        outputs = self.LLM.generate(**inputs, max_new_tokens=self.args.max_new_tokens, 
                                    num_return_sequences=1, 
                                    temperature=self.args.temperature,
                                    top_p=self.args.top_p,
                                    return_dict_in_generate=True, 
                                    output_scores=True,
                                    output_hidden_states=True,
                                    # length_penalty=self.args.length_penalty,
                                    # num_beams=self.args.num_beams,
                                    # do_sample=True
                                )
        
        logit_log_softmax, batch_loss = self.get_logits_and_loss(outputs, label)

        for index, (input, output, answer) in enumerate(zip(inputs["input_ids"], outputs["sequences"], batch_answer)):
            generation = self.LLM_tokenizer.decode(output, skip_special_tokens=True)
            pred, id_pred, hallucination_cnt = self.pasrse_record_res(my_input_list[index], label[index], generation, answer, training_flag, record_flag)
            batch_pred.append(pred)
            batch_id_pred.append(id_pred)
            batch_hallucination_cnt+=hallucination_cnt

        return batch_pred, batch_id_pred, batch_hallucination_cnt, save_doc_num, batch_loss, logit_log_softmax

    def get_logits_and_loss_1(self, outputs, label):
        token_hidden_list = []
        for token_rep in outputs["hidden_states"]: 
            token_hidden_list.append(token_rep[-1][:, -1, :])
        
        last_hidden_states = torch.stack(token_hidden_list, dim=1)
        logit = self.LLM.lm_head(last_hidden_states)
        logit_log_softmax = F.log_softmax(logit, dim=-1)/self.args.temperature

        generation_len = logit.size(1)
        label_len = max(len(seq) for seq in label)
        max_len = max(generation_len, label_len)
        label, loss_logit_log_softmax, label_mask = left_pad_loss_logit(label, logit_log_softmax, max_len, self.LLM_tokenizer.added_tokens_encoder[self.LLM_tokenizer.pad_token])

        loss_fct = nn.NLLLoss(reduction="none") 
        loss_logit_log_softmax = loss_logit_log_softmax.permute(0,2,1)
        # loss_logit_log_softmax = loss_logit_log_softmax.view(-1, self.args.num_beams, loss_logit_log_softmax.size(1), loss_logit_log_softmax.size(2))[:, 0, :, :]
        batch_loss = loss_fct(loss_logit_log_softmax, label) * label_mask

        logit_log_softmax = torch.mean(logit_log_softmax, dim=1) if logit_log_softmax.ndim==3 else logit_log_softmax
        batch_loss = torch.mean(batch_loss, dim=-1) if batch_loss.ndim==2 else batch_loss

        return logit_log_softmax, batch_loss

    def get_logits_and_loss(self, outputs, label):
        last_hidden_states = outputs["hidden_states"][0][-1]
        logit = self.LLM.lm_head(last_hidden_states)[:, -1, :]
        logit_log_softmax = F.log_softmax(logit, dim=-1)

        label = torch.LongTensor(label).to(logit_log_softmax.device)
        loss_fct = nn.NLLLoss(reduction="none")
        batch_loss = loss_fct(logit_log_softmax, label.view(-1))

        return logit_log_softmax, batch_loss
    
    def pasrse_record_res(self, my_input, label, generation, answer, training_flag, record_flag):
        pred, id_pred, hallucination_cnt = extracted_token_id_label(generation, label, self.LLM_tokenizer, self.args.dataset, self.prompt, self.args.LLM)

        if training_flag:
            result_logger = self.train_result_logger
        else:    
            result_logger = self.test_result_logger

        if record_flag:
            result_logger.info(f"my_input: {my_input}")
            result_logger.info(f"answer:   {answer} ")
            result_logger.info(f"pred:   {pred} ")
            result_logger.info(f"=================================================================================================================================================================================================\n\n")
            # result_logger.info(f"label:   {[self.LLM_tokenizer._convert_id_to_token(int(label_i))   for label_i in label] } ")
            # result_logger.info(f"id_pred: {[self.LLM_tokenizer._convert_id_to_token(id_pred_i) for id_pred_i in id_pred] } "+ "\n========================================================================================================================")
        return pred, id_pred, hallucination_cnt
    

   