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
from transformers import AutoTokenizer, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers.pipelines.pt_utils import KeyDataset
from dataloader.usmle_loader import Prompt_Dataset
import json
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils.metrics import My_Metrics
from torch.utils.tensorboard import SummaryWriter
from utils.utils import extracted_label, LineListOutputParser


class My_Trainer:

    def __init__(self, args, my_model, LLM, LLM_tokenizer, retri_encoder, triever_tokenizer, device):
        self.args = args
        self.print_logger = args.print_logger
        self.result_logger = args.result_logger

        self.device = device
        self.my_model = my_model

        self.LLM = LLM
        self.LLM_tokenizer = LLM_tokenizer
        self.my_metrics = My_Metrics()
        # self.writer = SummaryWriter(args.dir_path+"/runs/")
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
                
                break
            break
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

    def retrieve(self, query):

        query_input = self.triever_tokenizer(query, truncation=True, return_tensors='pt', max_length=self.args.chunk_size, padding=True).to(self.device)
        query_emb = self.query_encoder(**query_input).last_hidden_state[:, 0, :]

        scores = query_emb @ self.vectordb
        select_index = torch.argsort(scores, descending=True)[:, :self.args.max_retri_num].tolist()
        batch_doc = []
        for docs in select_index:
            retrieve_doc = [self.contexts[i] for i in docs]
            
            tmp_str = ""
            tmp_len_list = []
            if len(retrieve_doc)>0:
                for index, i in enumerate(retrieve_doc):
                    tmp_str += "document ("+ str(index) + ") \n\n"
                    tmp_str = tmp_str + i + "\n\n"
                    tmp_len_list.append(len(i))
            batch_doc.append(tmp_str)
            self.print_logger.info(f"retrieve document num: {len(retrieve_doc)}, length: {str(tmp_len_list)}")

            # self.result_logger.info(f"retrieve document: \n{tmp_str} \n")
        return batch_doc, scores

    def train_proc(self, train_data_loader, dev_data_loader):
        if not self.args.if_RA:
            raise Exception("need retrieve ! ")

        self.updata_retri_embedding()
        self.print_logger.info("Start training ... \n ")

        all_test_labels = []
        all_test_predictions = []
        
        cnt = 0
        for index, data_item in enumerate(train_data_loader):
            self.print_logger.info(f"process num: {index}")
            question = data_item['question']
            options = data_item['options']
            label = data_item["label"]
            
            my_input_list = []
            retrieve_doc, retrieve_scores = self.retrieve(question)

            if self.args.demonstration:
                demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                input_dict = {'question': question, 'options': options, "context": retrieve_doc, "demonstration": demonstration}
            else:
                input_dict = {'question': question, 'options': options, "context": retrieve_doc}
                
            # input = F.log_softmax(scores/tau1, dim=0)
            # target = tau2
            # kl_loss = self.kl_loss(input, target)
            all_test_predictions, all_test_labels, scores = self.return_prediction(input_dict, label, all_test_predictions, all_test_labels)

            # cnt+=1
            # if cnt ==10:
            #     break

        self.my_metrics.metrics_task_res(all_test_labels, all_test_predictions, self.args.print_logger)
            

    def test_proc(self, test_data_loader, dev_data_loader):
        if self.args.if_RA:
            self.updata_retri_embedding()
            
        self.print_logger.info("Start test ... \n ")

        all_test_labels = []
        all_test_predictions = []
        
        cnt = 0
        for index, data_item in enumerate(test_data_loader):
            self.print_logger.info(f"process num: {index}")
            question = data_item['question']
            options = data_item['options']
            label = data_item["label"]

            if self.args.if_RA:
                with torch.no_grad():
                    retrieve_doc, retrieve_scores = self.retrieve(question)
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
                all_test_predictions, all_test_labels, scores = self.return_prediction(input_dict, label, all_test_predictions, all_test_labels)

            # cnt+=1
            # if cnt ==10:
            #     break

        self.my_metrics.metrics_task_res(all_test_labels, all_test_predictions, self.args.print_logger)

    def return_prediction(self, input_dict, label, all_test_predictions, all_test_labels):
        my_input_list = []
        keys = input_dict.keys()
        for values in zip(*input_dict.values()):
            current_inputs = dict(zip(keys, values))
            my_input = self.prompt.format(**current_inputs)
            my_input_list.append(my_input)

        dataset = Prompt_Dataset(my_input_list)
        for index, generation in enumerate(self.pipe(dataset)):
            pred = self.pasrse_record_res(my_input_list[index], label[index], generation[0]["generated_text"][-self.args.max_new_tokens:])
            all_test_predictions.append(pred)
            all_test_labels.append(label[index])
        return all_test_predictions, all_test_labels, generation[0]["scores"]
    
    def pasrse_record_res(self, my_input, label, generation):
        pred = extracted_label(generation)

        self.result_logger.info(my_input)
        self.result_logger.info(generation)
        self.result_logger.info(f"label: {label}")
        self.result_logger.info(f"pred: {pred} "+ "\n========================================================================================================================")
        
        return pred