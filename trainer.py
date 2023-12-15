from langchain.vectorstores import Chroma

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import EmbeddingsRedundantFilter
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dataloader.usmle_loader import Prompt_Dataset
import torch.nn.functional as F
import json
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import My_Metrics
from torch.utils.tensorboard import SummaryWriter
from utils.utils import extracted_label, map_prob, LineListOutputParser, empty_logger_file


class My_Trainer:

    def __init__(self, args, my_model, LLM, LLM_tokenizer, device, retri_encoder=None, triever_tokenizer=None, all_retrieve_doc=None, text_splitter=None):
        self.args = args
        self.print_logger = args.print_logger
        self.test_result_logger = args.test_result_logger
        self.train_result_logger = args.train_result_logger

        self.device = device
        self.my_model = my_model

        self.LLM = LLM
        self.LLM_tokenizer = LLM_tokenizer
        self.my_metrics = My_Metrics()
        self.writer = SummaryWriter(args.dir_path+"/runs/")
        if args.LLM != "chatGPT":
            self.pipe = pipeline(
                    "text-generation",
                    model=LLM,
                    tokenizer=self.LLM_tokenizer,
                    # max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                    device_map="auto",
                    batch_size=self.args.test_batch_size, 
                    output_scores=True, return_dict_in_generate=True,
                ) 
        else:
            self.pipe = LLMChain(prompt=self.prompt, llm=LLM)

        if self.args.if_train:
            self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            # self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        if self.args.if_RA:
            self.retrieved_document = all_retrieve_doc
            self.text_splitter = text_splitter
            self.triever_tokenizer =  triever_tokenizer

            if args.triever in ["dragon+"]:
                self.retriever =  retri_encoder.to(self.device)
            else:
                raise Exception("wrong !")
            
            if self.args.retriever_ckpt_path is not None:
                self.retriever.load_state_dict(torch.load(self.args.retriever_ckpt_path))
                self.print_logger.info(f"load retriever: {self.args.retriever_ckpt_path} \n")    

            if self.args.if_train:
                param_list = list(self.retriever.parameters())
                self.optimizer = torch.optim.Adam( param_list, lr=self.args.lr, weight_decay=self.args.l2_coef)

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
    
    def get_retriever(self):
        # self.vectordb._embedding_function = self.embeddings_query_fn
        retriever=self.vectordb.as_retriever(search_kwargs={"k": self.args.infer_retri_num})

        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings_fn)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings_fn, similarity_threshold=self.args.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(transformers=[self.text_splitter,  redundant_filter, relevant_filter])

        if self.args.multi_query:
            output_parser = LineListOutputParser()
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate five 
                different versions of the given user question to retrieve relevant documents from a vector 
                database. By generating multiple perspectives on the user question, your goal is to help
                the user overcome some of the limitations of the distance-based similarity search. 
                Provide these alternative questions separated by newlines.
                Original question: {question}""" )

            llm_chain = LLMChain(llm= ChatOpenAI(temperature=0), prompt=QUERY_PROMPT, output_parser=output_parser)
            retriever = MultiQueryRetriever(retriever=retriever, llm_chain=llm_chain, parser_key="lines")

        retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
        return retriever
    
    def get_llm_likelihoods(self, train_data_loader):
        self.updata_retri_embedding()

        self.print_logger.info(f"len of train_data_loader : {len(train_data_loader)}")
        res_dic = {}
        step_num = -1
        for data_item in train_data_loader:
            step_num+=1
            self.print_logger.info(f"step_num : {step_num}")
            question = data_item['question']
            options = data_item['options']
            batch_label = data_item["label"]
            

            query_input = self.triever_tokenizer(question, truncation=True, return_tensors='pt', max_length=self.args.chunk_size, padding=True).to(self.device)
            query_emb = self.retriever(**query_input).last_hidden_state[:, 0, :]

            scores = query_emb @ self.vectordb
            batch_retrieve_index = torch.argsort(scores, descending=True)[:, :self.args.preprocess_retri_num].tolist()
            retrieve_docs = []
            for docs in batch_retrieve_index:
                retrieve_doc = [self.retrieved_document[i] for i in docs]
                retrieve_docs.append(retrieve_doc)

            transposed_retrieve_docs = list(map(list, zip(*retrieve_docs))) # (preprocess_retri_num, train_batch_size)
            transposed_retrieve_index = list(map(list, zip(*batch_retrieve_index))) # (preprocess_retri_num, train_batch_size)

            for one_batch_retrieve_doc, doc_index in zip(transposed_retrieve_docs, transposed_retrieve_index):
                input_dict = {'question': question, 'options': options, "context": one_batch_retrieve_doc}
                batch_pred, scores = self.pipeline_inference(input_dict, batch_label, training_flag=True)
                llm_likelihoods = map_prob(batch_label, scores, self.LLM_tokenizer)
                
                for ques, llm_likelihood, retrieve_index in zip(question, llm_likelihoods, doc_index):
                    res_dic.setdefault(ques, []) 
                    res_dic[ques].append([retrieve_index, float(llm_likelihood)])

        with open(str(self.args.LLM)+str(self.args.preprocess_retri_num)+".json", "w", encoding='utf-8') as f:
            json.dump(res_dic, f)

    def updata_retri_embedding(self):
        self.print_logger.info("updata_retri_embedding ...")
        start_time = time.time()
        with torch.no_grad():

            context_batches = [self.retrieved_document[i:i + self.args.retri_batch_size] for i in range(0, len(self.retrieved_document), self.args.retri_batch_size)]

            vectordb = []
            for context in tqdm(context_batches):
                ctx_input = self.triever_tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=self.args.chunk_size).to(self.device)
                tmp_res = self.retriever(**ctx_input).last_hidden_state[:, 0, :]
                vectordb.append(tmp_res.detach().cpu().to(self.device))
                torch.cuda.empty_cache()

            self.vectordb  = torch.cat((vectordb), dim=0).transpose(0, 1)
            self.print_logger.info(f"vectordb size: {self.vectordb.size()}")
            self.print_logger.info("updata_retri_embedding in %.2f sec. \n"% (time.time() - start_time))

    def retrieve(self, query, retri_num, train_flag):
        if train_flag and retri_num==1:
            raise Exception("train need retrieve more than one docs !")
        
        query_input = self.triever_tokenizer(query, truncation=True, return_tensors='pt', max_length=self.args.chunk_size, padding=True).to(self.device)
        query_emb = self.retriever(**query_input).last_hidden_state[:, 0, :]

        scores = query_emb @ self.vectordb
        batch_select_index = torch.argsort(scores, descending=True)[:, :retri_num].tolist()
        batch_infer_doc = []
        for docs in batch_select_index:
            retrieve_doc = [self.retrieved_document[i] for i in docs]

            batch_infer_doc.append(retrieve_doc)

        return batch_infer_doc, scores
    
    def compute_lsr_loss(self, retrieve_scores, batch_llm_score):
        input = F.softmax(retrieve_scores/self.args.retrieval_tau, dim=-1)
        # target = F.log_softmax(batch_llm_score/self.args.llm_tau, dim=-1).to(input.device)
        target = F.softmax(batch_llm_score/self.args.llm_tau, dim=-1).to(input.device)
        lsr_loss = self.kl_loss(input, target)
        return lsr_loss

    def train_proc(self, train_data_loader, dev_data_loader, test_data_loader):
        if not self.args.if_RA:
            raise Exception("need retrieve ! ")

        self.updata_retri_embedding()
        self.print_logger.info("Start training ... \n ")
        
        # test_acc, test_precision, test_recall, test_f1 = self.test_proc(test_data_loader, dev_data_loader)
        # self.writer.add_scalar('Performance/test/acc', test_acc, 0)
        # self.writer.add_scalar('Performance/test/precision', test_precision, 0)
        # self.writer.add_scalar('Performance/test/recall', test_recall, 0)
        # self.writer.add_scalar('Performance/test/f1', test_f1, 0)

        all_train_labels = []
        all_train_predictions = []
        total_batch = len(train_data_loader)
        step_num = -1
        best_step =0
        best_acc =0
        for _ in range(self.args.epoch):
            for data_item in train_data_loader:
                step_num+=1
                question = data_item['question']
                options = data_item['options']
                batch_label = data_item["label"]
                
                retrieve_docs, retrieve_scores = self.retrieve(question, self.args.train_retri_num, train_flag=True)
                
                batch_llm_score = []
                batch_pred_list = []

                transposed_retrieve_docs = list(map(list, zip(*retrieve_docs))) # (train_retri_num, train_batch_size)
                for batch_retrieve_doc in transposed_retrieve_docs:
                    if self.args.demonstration:
                        demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                        input_dict = {'question': question, 'options': options, "context": batch_retrieve_doc, "demonstration": demonstration}
                    else:
                        input_dict = {'question': question, 'options': options, "context": batch_retrieve_doc}
            
                        batch_pred, scores = self.pipeline_inference(input_dict, batch_label, training_flag=True)
                        llm_likelihood = map_prob(batch_label, scores, self.LLM_tokenizer)
                        batch_llm_score.append(llm_likelihood)
                        batch_pred_list.append(batch_pred)
                
                lsr_loss = self.compute_lsr_loss(retrieve_scores[:, :self.args.train_retri_num],  torch.stack(batch_llm_score).permute(1,0))
                total_loss = lsr_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Loss/total_loss', total_loss, step_num)
                self.print_logger.info(f"training process num: {step_num}/{total_batch}, total_loss: {total_loss}, best_step :{best_step}, best_acc :{best_acc}")

                # only choose the most likely doc to pred in training stage
                all_train_predictions+=batch_pred_list[0]
                all_train_labels+=batch_label

                if (step_num % self.args.train_eval==0) and step_num>1:
                        
                    self.train_result_logger = empty_logger_file(self.train_result_logger)
                    train_acc, train_precision, train_recall, train_f1 = self.my_metrics.metrics_task_res(all_train_labels, 
                                                                        all_train_predictions, self.args.print_logger, "train")

                    self.writer.add_scalar('Performance/train/acc', train_acc, step_num)
                    self.writer.add_scalar('Performance/train/precision', train_precision, step_num)
                    self.writer.add_scalar('Performance/train/recall', train_recall, step_num)
                    self.writer.add_scalar('Performance/train/f1', train_f1, step_num)

                    # test_acc, test_precision, test_recall, test_f1 = self.test_proc(test_data_loader, dev_data_loader, break_cnt=500)
                    test_acc, test_precision, test_recall, test_f1 = self.test_proc(test_data_loader, dev_data_loader)

                    self.writer.add_scalar('Performance/test/acc', test_acc, step_num)
                    self.writer.add_scalar('Performance/test/precision', test_precision, step_num)
                    self.writer.add_scalar('Performance/test/recall', test_recall, step_num)
                    self.writer.add_scalar('Performance/test/f1', test_f1, step_num)

                    self.updata_retri_embedding()

                    all_train_labels = []
                    all_train_predictions = []

                    if test_acc>best_acc:
                        best_acc = test_acc
                        best_step = step_num
                        if step_num>10:
                            torch.save(self.retriever.state_dict(), self.args.dir_path+'/retriever.pkl') 
                
            #     step_num+=1
            #     if step_num ==10:
            #         break
            # if step_num ==10:
            #   break

    def test_proc(self, test_data_loader, dev_data_loader, break_cnt=None):
        if self.args.if_RA and (self.args.if_train is False):
            self.updata_retri_embedding()
            
        self.print_logger.info("Start test ... \n ")

        all_test_labels = []
        all_test_predictions = []
        
        for index, data_item in enumerate(test_data_loader):
            # if index%100==0:
            self.print_logger.info(f"testing process num: {index}")
            question = data_item['question']
            options = data_item['options']
            batch_label = data_item["label"]

            if self.args.if_RA:
                with torch.no_grad():
                    retrieve_doc, _ = self.retrieve(question, self.args.infer_retri_num, train_flag=False)
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': question, 'options': options, "context": retrieve_doc, "demonstration": demonstration}
                else:
                    input_dict = {'question': question, 'options': options, "context": retrieve_doc}
            else:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': question, 'options': options, "demonstration": demonstration}
                else:
                    input_dict = {'question': question, 'options': options}
            
            with torch.no_grad():
                batch_pred, _ = self.pipeline_inference(input_dict, batch_label)

            all_test_labels+=batch_label
            all_test_predictions+=batch_pred
            if break_cnt is not None and break_cnt<index:
                break

        test_acc, test_precision, test_recall, test_f1 = self.my_metrics.metrics_task_res(all_test_labels, all_test_predictions, self.args.print_logger, "test")
        return test_acc, test_precision, test_recall, test_f1
    
    def pipeline_inference(self, input_dict, labels, training_flag=False):
        my_input_list = []
        new_label = []
        for label in labels:
            for j in range(self.args.infer_retri_num):
                new_label.append(label)

        for values in zip(*input_dict.values()):
            for each_doc in values[-1]:
                my_input = self.prompt.format(question=values[0], options=values[1], context=each_doc)
                my_input_list.append(my_input)

        batch_predictions = []
        batch_score = []
        dataset = Prompt_Dataset(my_input_list)
        tmp_tesnor = 0
        for index, generation in enumerate(self.pipe(dataset)):
            res_tensor = generation[0]["scores"].squeeze()
            tensor = torch.where(res_tensor == float('-inf'), torch.tensor(1e-10), res_tensor)
            tmp_tesnor+=tensor
            if index>1 and index%(self.args.infer_retri_num-1)==0:
                tmp_score = tmp_tesnor/self.args.infer_retri_num
                pred = self.LLM_tokenizer._convert_id_to_token(int(torch.argmax(tmp_score)))
                pred = extracted_label(pred)
                batch_predictions.append(pred)
                batch_score.append(tmp_score)
                tmp_tesnor = 0

        return batch_predictions, batch_score
    
    def pasrse_record_res(self, my_input, label, generation, training_flag):
        pred = extracted_label(generation)

        if training_flag:
            result_logger = self.train_result_logger
        else:    
            result_logger = self.test_result_logger

        result_logger.info(my_input)
        result_logger.info(generation)
        result_logger.info(f"label: {label}")
        result_logger.info(f"pred: {pred} "+ "\n========================================================================================================================")
        
        return pred
    