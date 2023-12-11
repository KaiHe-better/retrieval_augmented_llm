from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from utils.utils import extracted_label, map_prob, LineListOutputParser


class My_Trainer:

    def __init__(self, args, my_model, LLM, LLM_tokenizer, retri_encoder, triever_tokenizer, device):
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
            self.chunk_file_path =  os.path.join(args.retrieval_processed_file_dir, str(args.triever)+"_"+str(args.chunk_size)+"_"+str(args.chunk_overlap)+"_str.txt"  )
            self.triever_tokenizer =  triever_tokenizer
            self.retriever_embedding = []
            self.retriever_txt = []
            self.retrieved_document, self.text_splitter = self.process_document()

            if args.triever in ["dragon+"]:
                self.query_encoder =  retri_encoder[0].to(self.device)
                self.context_encoder = retri_encoder[1].to(self.device)
            else:
                raise Exception("wrong !")
            
            if self.args.if_train:
                param_list = []
                for name, param in self.query_encoder.named_parameters():
                    param_list.append(param)
                for name, param in self.context_encoder.named_parameters():
                    param_list.append(param)
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
        
    def process_document(self):
        start_time = time.time()
        if not os.path.exists(self.args.retrieval_processed_file_dir):
            os.makedirs(self.args.retrieval_processed_file_dir)
        
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.triever_tokenizer, chunk_size=self.args.chunk_size, chunk_overlap=self.args.chunk_overlap)

        all_doc = []
        self.print_logger.info("chunk retrieval files ... \n")
        for root, dirs, files in os.walk(self.args.retrieval_raw_data_dir):
            for file in files:
                loader = TextLoader(os.path.join(root, file))
                documents = loader.load()
                chunks = text_splitter.split_documents(documents)
                all_doc += chunks
                
                if self.args.test_code_flag:
                    break
            if self.args.test_code_flag:
                break
        if self.args.test_code_flag:
            self.print_logger.info("===============================breaking ===============================")
            self.print_logger.info("===============================breaking ===============================")
            self.print_logger.info("===============================breaking ===============================")

        self.print_logger.info("process retrieval files finish in %.2f sec. \n"% (time.time() - start_time))
        return all_doc, text_splitter

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
        retriever=self.vectordb.as_retriever(search_kwargs={"k": self.args.max_retri_num})

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
                Original question: {question}""",
            )

            llm_chain = LLMChain(llm= ChatOpenAI(temperature=0), prompt=QUERY_PROMPT, output_parser=output_parser)
            retriever = MultiQueryRetriever(retriever=retriever, llm_chain=llm_chain, parser_key="lines")

        retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
        return retriever
    
    def updata_retri_embedding(self):
        self.print_logger.info("updata_retri_embedding ...")
        start_time = time.time()
        with torch.no_grad():
            self.contexts = [i.page_content for i in self.retrieved_document]

            context_batches = [self.contexts[i:i + self.args.retri_batch_size] for i in range(0, len(self.contexts), self.args.retri_batch_size)]

            vectordb = []
            for context in tqdm(context_batches):
                ctx_input = self.triever_tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=self.args.chunk_size).to(self.device)
                tmp_res = self.context_encoder(**ctx_input).last_hidden_state[:, 0, :]
                vectordb.append(tmp_res.detach().cpu().to(self.device))
                torch.cuda.empty_cache()

            self.vectordb  = torch.cat((vectordb), dim=0).transpose(0, 1)
            self.print_logger.info(f"vectordb size: {self.vectordb.size()}")
            self.print_logger.info("updata_retri_embedding in %.2f sec. \n"% (time.time() - start_time))

    def retrieve(self, query, max_retri_num, train_flag):
        if train_flag and max_retri_num==1:
            raise Exception("train need retrieve more than one docs !")
        
        query_input = self.triever_tokenizer(query, truncation=True, return_tensors='pt', max_length=self.args.chunk_size, padding=True).to(self.device)
        query_emb = self.query_encoder(**query_input).last_hidden_state[:, 0, :]

        scores = query_emb @ self.vectordb
        batch_select_index = torch.argsort(scores, descending=True)[:, :max_retri_num].tolist()
        batch_infer_doc = []
        for docs in batch_select_index:
            retrieve_doc = [self.contexts[i] for i in docs]

            if train_flag:
                batch_infer_doc.append(retrieve_doc)
            else:
                tmp_str = ""
                tmp_len_list = []
                if len(retrieve_doc)>0:
                    for index, i in enumerate(retrieve_doc):
                        tmp_str += "document ("+ str(index) + ") \n\n"
                        tmp_str = tmp_str + i + "\n\n"
                        tmp_len_list.append(len(i))
                batch_infer_doc.append(tmp_str)
                # self.print_logger.info(f"retrieve document num: {len(retrieve_doc)}, length: {str(tmp_len_list)}")

        return batch_infer_doc, scores
    
    def compute_lsr_loss(self, retrieve_scores, batch_llm_score):
        input = F.log_softmax(retrieve_scores/self.args.retrieval_tau, dim=-1)
        target = F.log_softmax(batch_llm_score/self.args.llm_tau, dim=-1).to(input.device)
        lsr_loss = -self.kl_loss(input, target)
        return lsr_loss

    def train_proc(self, train_data_loader, dev_data_loader):
        if not self.args.if_RA:
            raise Exception("need retrieve ! ")

        self.updata_retri_embedding()
        self.print_logger.info("Start training ... \n ")
        
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
                
                retrieve_docs, retrieve_scores = self.retrieve(question, self.args.max_train_retri_num, train_flag=True)
                
                batch_llm_score = []
                batch_pred_list = []

                transposed_retrieve_docs = list(map(list, zip(*retrieve_docs))) # (max_train_retri_num, train_batch_size)
                for batch_retrieve_doc in transposed_retrieve_docs:
                    if self.args.demonstration:
                        demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                        input_dict = {'question': question, 'options': options, "context": batch_retrieve_doc, "demonstration": demonstration}
                    else:
                        input_dict = {'question': question, 'options': options, "context": batch_retrieve_doc}
            
                        batch_pred, scores = self.return_prediction(input_dict, batch_label, training_flag=True)
                        llm_likelihood = map_prob(batch_label, scores, self.LLM_tokenizer)
                        batch_llm_score.append(llm_likelihood)
                        batch_pred_list.append(batch_pred)

                lsr_loss = self.compute_lsr_loss(retrieve_scores[:, :self.args.max_train_retri_num],  torch.stack(batch_llm_score).permute(1,0))
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
                    file_path = self.train_result_logger.handlers[0].baseFilename
                    if os.path.isfile(file_path):
                        pass

                    acc, precision, recall, f1 = self.my_metrics.metrics_task_res(all_train_labels, all_train_predictions, self.args.print_logger)
                    
                    self.writer.add_scalar('Performance/acc', acc, step_num)
                    self.writer.add_scalar('Performance/precision', precision, step_num)
                    self.writer.add_scalar('Performance/recall', recall, step_num)
                    self.writer.add_scalar('Performance/f1', f1, step_num)

                    all_train_labels = []
                    all_train_predictions = []

                    if acc>best_acc:
                        best_acc = acc
                        best_step = step_num
                        if step_num>10:
                            torch.save(self.query_encoder.state_dict(), self.args.dir_path+'/query_encoder.pkl') 
                            torch.save(self.context_encoder.state_dict(), self.args.dir_path+'/context_encoder.pkl') 

            #     step_num+=1
            #     if step_num ==10:
            #         break
            # if step_num ==10:
            #   break

    def test_proc(self, test_data_loader, dev_data_loader):
        if self.args.if_RA:
            self.updata_retri_embedding()
            
        self.print_logger.info("Start test ... \n ")

        all_test_labels = []
        all_test_predictions = []
        
        cnt = 0
        for index, data_item in enumerate(test_data_loader):
            self.print_logger.info(f"testing process num: {index}")
            question = data_item['question']
            options = data_item['options']
            batch_label = data_item["label"]

            if self.args.if_RA:
                with torch.no_grad():
                    retrieve_doc, _ = self.retrieve(question, self.args.max_retri_num, train_flag=False)
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
                batch_pred, _ = self.return_prediction(input_dict, batch_label)

            all_test_labels.append(batch_label)
            all_test_predictions.append(batch_pred)

            # cnt+=1
            # if cnt ==10:
            #     break

        acc, precision, recall, f1 = self.my_metrics.metrics_task_res(all_test_labels, all_test_predictions, self.args.print_logger)

    def return_prediction(self, input_dict, label, training_flag=False):
        my_input_list = []
        keys = input_dict.keys()
        for values in zip(*input_dict.values()):
            current_inputs = dict(zip(keys, values))
            my_input = self.prompt.format(**current_inputs)
            my_input_list.append(my_input)

        batch_predictions = []
        batch_score = []
        dataset = Prompt_Dataset(my_input_list)
        for index, generation in enumerate(self.pipe(dataset)):
            pred = self.pasrse_record_res(my_input_list[index], label[index], generation[0]["generated_text"][-self.args.max_new_tokens:], training_flag)
            batch_predictions.append(pred)
            batch_score.append(generation[0]["scores"].squeeze())
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
    