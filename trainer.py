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
import json
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils.metrics import My_Metrics
from torch.utils.tensorboard import SummaryWriter
from utils.utils import extracted_label, LineListOutputParser


class My_Trainer:

    def __init__(self, args, LLM, LLM_tokenizer, retri_encoder_path, triever_tokenizer_path, device):
        self.args = args
        self.print_logger = args.print_logger
        self.result_logger = args.result_logger

        self.device = device

        self.LLM = LLM
        self.LLM_tokenizer = LLM_tokenizer
        self.my_metrics = My_Metrics()
        # self.writer = SummaryWriter(args.dir_path+"/runs/")
        if args.LLM != "chatGPT":
            self.pipe = pipeline(
                    "text-generation",
                    model=LLM,
                    tokenizer=self.LLM_tokenizer,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                    device_map="auto",
                ) 
            self.my_model_pipeline = HuggingFacePipeline(pipeline=self.pipe)
        else:
            self.my_model_pipeline = LLM

        if self.args.if_RA:
            self.chunk_file_path =  os.path.join(args.retrieval_processed_file_dir, str(args.triever)+"_"+str(args.chunk_size)+"_"+str(args.chunk_overlap)+"_str.txt"  )
            self.triever_tokenizer =  AutoTokenizer.from_pretrained(triever_tokenizer_path)
            self.retriever_embedding = []
            self.retriever_txt = []
            self.retrieved_document, self.text_splitter = self.process_document()
            if args.triever in ["dragon+"]:
                self.query_encoder =  AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').to(self.device)
                self.context_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').to(self.device)

            else:
                self.embeddings_fn = HuggingFaceEmbeddings(model_name=retri_encoder_path, 
                                                        model_kwargs = {'device': self.device, }, 
                                                        encode_kwargs = {'normalize_embeddings': False, "batch_size":self.args.retri_batch_size } )

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

        self.demon_prompt = PromptTemplate.from_template("demon_prompt")
        self.prompt = PromptTemplate.from_template(prompt_text)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.my_model_pipeline)
            
    def pasrse_record_res(self, label, generation):
        self.result_logger.info(generation)
        pred = extracted_label(generation)

        self.result_logger.info(f"label: {label}")
        self.result_logger.info(f"pred: {pred} "+ "\n========================================================================================================================")
        
        return pred

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
                
        #         break
        #     break
        # self.print_logger.info("===============================breaking ===============================")
        # self.print_logger.info("===============================breaking ===============================")
        # self.print_logger.info("===============================breaking ===============================")

        self.print_logger.info("process retrieval files finish in %.2f sec. \n"% (time.time() - start_time))
        return all_doc, text_splitter

    def random_select_demonstration(self, train_data_loader):
        demon_prompt = ""
        for index, item in enumerate(train_data_loader):
            demon_prompt += "Demonstration " + str(index)+"\n Question: {} \n Options: {} \n Answer: <{}> \n\n".format(item["question"][0], item["options"][0], item["answer"][0])
        self.result_logger.info(f" \n {demon_prompt} \n")
        return demon_prompt
    
    def get_retriever(self):
        # self.vectordb._embedding_function = self.embeddings_query_fn
        retriever=self.vectordb.as_retriever(search_kwargs={"k": self.args.max_retri_num})

        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings_context_fn)
        relevant_filter = EmbeddingsFilter(embeddings=self.embeddings_context_fn, similarity_threshold=self.args.similarity_threshold)
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
    
    def updata_retri_embedding_111(self):
        self.print_logger.info("updata_retri_embedding ...")
        start_time = time.time()
        with torch.no_grad():
            self.vectordb = Chroma.from_documents(self.retrieved_document, self.embeddings_context_fn)  
        self.print_logger.info("updata_retri_embedding in %.2f sec. \n"% (time.time() - start_time))

    def retrieve_111(self, query):
        retrieve_doc = self.retriever.get_relevant_documents(query=query)
        tmp_str = ""
        tmp_len_list = []
        if len(retrieve_doc)>0:
            for index, i in enumerate(retrieve_doc):
                tmp_str += "document ("+ str(index) + ") \n\n"
                tmp_str = tmp_str + i.page_content + "\n\n"
                tmp_len_list.append(len(i.page_content))

        self.print_logger.info(f"retrieve document num: {len(retrieve_doc)}, length: {str(tmp_len_list)}")
        self.result_logger.info(f"retrieve document: \n{tmp_str} \n")
        return tmp_str



    def updata_retri_embedding(self):
        self.print_logger.info("updata_retri_embedding ...")
        start_time = time.time()
        with torch.no_grad():
            self.contexts = [i.page_content for i in self.retrieved_document]

            context_batches = [self.contexts[i:i + self.args.retri_batch_size] for i in range(0, len(self.contexts), self.args.retri_batch_size)]

            vectordb = []
            for context in context_batches:
                ctx_input = self.triever_tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=self.args.chunk_size).to(self.device)
                tmp_res = self.context_encoder(**ctx_input).last_hidden_state[:, 0, :]
                vectordb.append(tmp_res )

        self.vectordb  = torch.cat((vectordb), dim=0).transpose(0, 1)
        self.print_logger.info(f"vectordb size: {self.vectordb.size()}")
        self.print_logger.info("updata_retri_embedding in %.2f sec. \n"% (time.time() - start_time))

    def retrieve(self, query):

        query_input = self.triever_tokenizer(query, return_tensors='pt').to(self.device)
        query_emb = self.query_encoder(**query_input).last_hidden_state[:, 0, :]
        
        query_emb = query_emb.unsqueeze(0)

        scores = (query_emb @ self.vectordb).squeeze()
        select_index = torch.argsort(scores, descending=True)[:self.args.max_retri_num].tolist()
        retrieve_doc = [self.contexts[i] for i in select_index]


        tmp_str = ""
        tmp_len_list = []
        if len(retrieve_doc)>0:
            for index, i in enumerate(retrieve_doc):
                tmp_str += "document ("+ str(index) + ") \n\n"
                tmp_str = tmp_str + i + "\n\n"
                tmp_len_list.append(len(i))

        self.print_logger.info(f"retrieve document num: {len(retrieve_doc)}, length: {str(tmp_len_list)}")
        self.result_logger.info(f"retrieve document: \n{tmp_str} \n")
        return tmp_str




    
    def train_proc(self, train_data_loader, dev_data_loader, test_data_loader):
        if self.args.if_RA:
            self.updata_retri_embedding()
            # self.retriever = self.get_retriever()

        self.print_logger.info("Start training... \n ")

        all_test_labels = []
        all_test_predictions = []
        
        cnt = 0
        for index, data_item in enumerate(test_data_loader):
            query = data_item['question'][0]
            options = data_item['options'][0]
            label = data_item["label"][0]

            self.print_logger.info(f"process num: {index}")
            self.result_logger.info(f"process num: {index}")
            self.result_logger.info(f"query: {query} \n")
            self.result_logger.info(f"options: {options} \n")

            if self.args.if_RA:
                retrieve_doc = self.retrieve(query)
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(train_data_loader)
                    generation = self.llm_chain.run(question=query, options=options, context=retrieve_doc, demonstration=demonstration)
                else:
                    generation = self.llm_chain.run(question=query, options=options, context=retrieve_doc)
            else:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(train_data_loader)
                    generation = self.llm_chain.run(question=query, options=options, demonstration=demonstration)
                else:
                    generation = self.llm_chain.run(question=query, options=options)
            pred = self.pasrse_record_res(label, generation)
            all_test_labels.append(label)
            all_test_predictions.append(pred)

            # cnt+=1
            # if cnt ==10:
            #     break


        acc, precision, recall, f1 = self.my_metrics.metrics_task_res(all_test_labels, all_test_predictions)

        self.args.print_logger.info(f"acc {round(acc*100, 2)}")
        self.args.print_logger.info(f"precision {round(precision*100, 2)}")
        self.args.print_logger.info(f"recall {round(recall*100, 2)}")
        self.args.print_logger.info(f"f1 {round(f1*100, 2)}")

