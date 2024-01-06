from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import EmbeddingsRedundantFilter
from transformers import pipeline
from langchain.chains import LLMChain
from dataloader.usmle_loader import Prompt_Dataset
import torch.nn.functional as F
import json
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import My_Metrics
from sklearn.metrics import  accuracy_score
from torch.utils.tensorboard import SummaryWriter
from utils.utils import extracted_label, map_prob, LineListOutputParser, empty_logger_file, combine_doc, __dist__, calculate_perplexity


class My_Trainer:

    def __init__(self, args, MI_learner, LLM, LLM_tokenizer, device, retri_encoder=None, triever_tokenizer=None, all_retrieve_doc=None, text_splitter=None):
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
                # param_list_retriever = list(self.retriever.parameters())
                param_list_MI_learner = list(self.MI_learner.parameters())
                self.optimizer = torch.optim.Adam( param_list_MI_learner, lr=self.args.lr, weight_decay=self.args.l2_coef)

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

    def updata_retri_embedding(self):
        self.print_logger.info("updata_retri_embedding ...")
        start_time = time.time()
        with torch.no_grad():

            context_batches = [self.retrieved_document[i:i + self.args.retri_batch_size] for i in range(0, len(self.retrieved_document), self.args.retri_batch_size)]

            vectordb = []
            for context in tqdm(context_batches):
                ctx_input = self.triever_tokenizer(context, padding=True, truncation=True, return_tensors='pt', max_length=self.args.chunk_size).to(self.device)
                # tmp_res = torch.mean(self.retriever(**ctx_input).last_hidden_state * ctx_input["attention_mask"][:, :, None], dim=1)
                tmp_res = self.retriever(**ctx_input).last_hidden_state[:, 0, :]
                vectordb.append(tmp_res.detach().cpu().to(self.device))
                torch.cuda.empty_cache()

            self.vectordb  = torch.cat((vectordb), dim=0).transpose(0, 1)
            self.print_logger.info(f"vectordb size: {self.vectordb.size()}")
            self.print_logger.info("updata_retri_embedding in %.2f sec. \n"% (time.time() - start_time))
    
    def retrieve(self, query, retri_num, train_flag):
        if train_flag and retri_num==1:
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

        scores = __dist__(query_emb, self.vectordb, method='dot')
        batch_select_index = torch.argsort(scores, descending=True)[:, :retri_num].tolist()

        # batch_select_index = torch.argsort(__dist__(query_emb, self.vectordb, method='dot'), descending=True)[:, :retri_num].tolist()
        # batch_select_index = torch.argsort(__dist__(query_emb, self.vectordb, method='euclidean'), descending=True)[:, :retri_num].tolist()
        # batch_select_index = torch.argsort(__dist__(query_emb, self.vectordb, method='cosine'), descending=True)[:, :retri_num].tolist()
        
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
            retrieve_doc = [self.retrieved_document[i] for i in docs]
            batch_item_list.append(retrieve_doc)
            tmp_str = combine_doc(retrieve_doc)        
            batch_infer_doc.append(tmp_str)

        return batch_infer_doc, batch_item_list, query_embs, query_input["attention_mask"]
    
    def compute_lsr_loss(self, retrieve_scores, batch_llm_score):
        input = F.log_softmax(retrieve_scores/self.args.retrieval_tau, dim=-1)

        tmp_prob = F.softmax(retrieve_scores, dim=-1) * batch_llm_score
        target = F.log_softmax( tmp_prob /self.args.llm_tau, dim=-1).to(input.device)

        # target[target<1e-5]=0
        lsr_loss = self.kl_loss(input, target)

        return lsr_loss

    def return_input_dict(self, dev_data_loader, question, options, retrieve_docs):
        if self.args.demonstration:
            demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
            input_dict = {'question': question, 'options': options, "context": retrieve_docs, "demonstration": demonstration}
        else:
            input_dict = {'question': question, 'options': options, "context": retrieve_docs}
        return input_dict
    
    def train_proc(self, train_data_loader, dev_data_loader, test_data_loader):
        if not self.args.if_RA:
            raise Exception("need retrieve ! ")

        self.updata_retri_embedding()
        self.print_logger.info("Start training ... \n ")
        
        total_batch = len(train_data_loader)
        step_num = -1
        best_step = 0
        best_acc = 0
        total_work_num =0
        enhanced_acc_list = []
        for epoch_num in range(self.args.epoch):
            for data_item in train_data_loader:
                step_num+=1
                question = data_item['question']
                answer = data_item['answer']
                options = data_item['options']
                labels = data_item['label']

                retrieve_docs, bags_list, query_emb, att_mask = self.retrieve(question, self.args.train_retri_num, train_flag=True)
                input_dict = self.return_input_dict(dev_data_loader, question, options, retrieve_docs)
                with torch.no_grad():
                    batch_pred,  old_batch_hallucination_cnt, save_doc_num, batch_loss = self.pipeline_inference(input_dict, labels, training_flag=True, record_flag=False)
                    bag_pesu_label = [1 if batch_pred[i] == labels[i] else 0 for i in range(min(len(batch_pred), len(labels)))]
                    old_acc = round(accuracy_score(labels, batch_pred), 2)

                for doc_index, doc_num in enumerate(save_doc_num):
                    bags_list[doc_index] = bags_list[doc_index][:doc_num]

                MSL_loss, new_retrieve_docs, select_doc_num, bag_pred = self.MI_learner(query_emb, att_mask, bags_list, bag_pesu_label, batch_loss, self.retriever, self.triever_tokenizer, True)
                # can accelerate by turn off this
                if self.args.confirm_enhanced_acc:
                    new_input_dict = self.return_input_dict(dev_data_loader, question, options, new_retrieve_docs)
                    new_batch_pred, batch_hallucination_cnt, _, _ = self.pipeline_inference(new_input_dict, labels, training_flag=True, record_flag=True)
                    enhanced_acc =  round(accuracy_score(labels, new_batch_pred), 2)
                    enhanced_acc_list.append(enhanced_acc)
                else:
                    batch_hallucination_cnt, enhanced_acc, new_batch_pred = 0, 0, 0 
                    enhanced_acc_list=[1]

                if enhanced_acc>old_acc:
                    total_work_num+=1
                if  enhanced_acc<old_acc:
                    total_work_num-=1

                total_loss = MSL_loss
                total_loss.backward()

                self.writer.add_scalar('Loss/total_loss', round(float(total_loss), 4), step_num)
                # self.print_logger.info(f"epoch_num:{epoch_num}, bag_pesu_label :{bag_pesu_label}, bag_pred:{bag_pred}, labels: {labels}, batch_pred:{batch_pred}, new_batch_pred:{new_batch_pred}, enhanced_acc:{enhanced_acc}, old_acc:{old_acc}")
                                       
                if (step_num + 1) % self.args.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.print_logger.info(f"epoch_num: {epoch_num}, training process num: {step_num}/{total_batch}, total_loss: {round(float(total_loss), 4)}, old_hall_cnt:{old_batch_hallucination_cnt}/{len(question)}, hall_cnt {batch_hallucination_cnt}/{len(question)}, select_doc_num: {select_doc_num}/{self.args.train_retri_num}, total_work_num: {total_work_num}, old_acc:{old_acc}, enhanced_acc:{round(sum(enhanced_acc_list)/len(enhanced_acc_list),2)}, best_step:{best_step}, best_acc: {best_acc}")

                if (step_num % self.args.train_eval==0) and step_num>1:
                    total_work_num = 0

                    self.train_result_logger = empty_logger_file(self.train_result_logger)

                    break_cnt = 2 if self.args.test_code_flag else None
                    test_acc, test_precision, test_recall, test_f1 = self.test_proc(test_data_loader, dev_data_loader, break_cnt=break_cnt)

                    self.writer.add_scalar('Performance/test/acc', test_acc, step_num)
                    self.writer.add_scalar('Performance/test/precision', test_precision, step_num)
                    self.writer.add_scalar('Performance/test/recall', test_recall, step_num)
                    self.writer.add_scalar('Performance/test/f1', test_f1, step_num)

                    if test_acc>best_acc:
                        best_acc = test_acc
                        best_step = step_num

                #         if step_num>10:
                #             torch.save(self.retriever.state_dict(), self.args.dir_path+'/retriever.pkl') 
                
                # if step_num % 1 ==0 :
                #     break
            # if step_num ==100:
            #   break

    def test_proc(self, test_data_loader, dev_data_loader, break_cnt=None):
        if self.args.if_RA and (self.args.if_train is False):
            self.updata_retri_embedding()
            
        self.print_logger.info("\n Start test ...  ")

        all_test_labels = []
        all_test_predictions = []
        total_hallucination_cnt = 0
        for index, data_item in enumerate(test_data_loader):
            if index%100==0:
                self.print_logger.info(f"testing process num: {index}")
            question = data_item['question']
            options = data_item['options']
            batch_label = data_item["label"]

            if self.args.if_RA:
                with torch.no_grad():
                    retrieve_docs, bags_list, query_emb, att_mask = self.retrieve(question, self.args.infer_retri_num, train_flag=False)
                    _, new_retrieve_docs, select_doc_num, _ = self.MI_learner(query_emb, att_mask, bags_list, "bag_pesu_label", "batch_loss", self.retriever, self.triever_tokenizer, False)
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': question, 'options': options, "context": new_retrieve_docs, "demonstration": demonstration}
                else:
                    input_dict = {'question': question, 'options': options, "context": new_retrieve_docs}
            else:
                if self.args.demonstration:
                    demonstration = self.random_select_demonstration(dev_data_loader, self.args.test_batch_size)
                    input_dict = {'question': question, 'options': options, "demonstration": demonstration}
                else:
                    input_dict = {'question': question, 'options': options}
          
            with torch.no_grad():
                batch_pred,  batch_hallucination_cnt, _, _ = self.pipeline_inference(input_dict, batch_label, training_flag=False, record_flag=True)

            all_test_labels+=batch_label
            all_test_predictions+=batch_pred
            total_hallucination_cnt += batch_hallucination_cnt
            if break_cnt is not None and break_cnt<index:
                break

        test_acc, test_precision, test_recall, test_f1 = self.my_metrics.metrics_task_res(all_test_labels, all_test_predictions)
        self.args.print_logger.info(f"test: acc {test_acc}, f1 {test_f1}, precision {test_precision}, recall {test_recall}, total_hallucination_cnt {total_hallucination_cnt}, hallucination {round(total_hallucination_cnt / len(all_test_predictions), 2)} \n ")

        return test_acc, test_precision, test_recall, test_f1
    
    def pipeline_inference(self, input_dict, label, training_flag=False, record_flag=True):
        if self.args.LLM == "chatGPT":
            batch_pred, batch_hallucination_cnt, save_doc_num = self.non_local_llm_infer(input_dict, label, training_flag, record_flag)
            batch_loss = 0
        else:
            batch_pred, batch_hallucination_cnt, save_doc_num, batch_loss= self.local_llm_infer(input_dict, label, training_flag, record_flag)

        return batch_pred, batch_hallucination_cnt, save_doc_num, batch_loss

    def non_local_llm_infer(self, input_dict, label, training_flag=False, record_flag=True):
        batch_pred = []
        keys = input_dict.keys()
        batch_hallucination_cnt = 0
        
        if training_flag:
            save_doc_num = [self.args.train_retri_num]*len(label)
        else:
            save_doc_num = [self.args.infer_retri_num]*len(label)

        for index2, values in enumerate(zip(*input_dict.values())):
            current_inputs = dict(zip(keys, values))
            try:
                pred = self.pipe(current_inputs)["text"][:2]
            except:
                current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:-1])
                self.print_logger.info("too long context, we short one retrieval results !")
                save_doc_num[index2] = save_doc_num[index2]-1
                try:
                    pred = self.pipe(current_inputs)["text"][:2]
                except:
                    current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:-1])
                    self.print_logger.info("too long context, we short two retrieval results !")
                    save_doc_num[index2] = save_doc_num[index2]-1
                    try:
                        pred = self.pipe(current_inputs)["text"][:2]
                    except:
                        current_inputs["context"] = "document ".join(current_inputs["context"].split("document")[:5])
                        self.print_logger.info("too long context for many times, we only take first 5 retrieval results !")
                        pred =  self.pipe(current_inputs)["text"][:2]
                        save_doc_num[index2] = 5

            pred, hallucination_cnt = self.pasrse_record_res( self.prompt.format(**current_inputs) , label[index2], pred, training_flag, record_flag) 
            batch_pred.append(pred)  
            batch_hallucination_cnt+=hallucination_cnt

        return batch_pred, batch_hallucination_cnt, save_doc_num
    
    def local_llm_infer(self, input_dict, label, training_flag=False, record_flag=True):
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
        
        batch_pred = []
        batch_score = []
        batch_hallucination_cnt = 0
        inputs = self.LLM_tokenizer(my_input_list, return_tensors="pt", padding=True).to(self.args.device)
        outputs = self.LLM.generate(**inputs, max_new_tokens=self.args.max_new_tokens, 
                                    num_return_sequences=1, 
                                    temperature=self.args.temperature,
                                    top_p=self.args.top_p,
                                    return_dict_in_generate=True, 
                                    # output_scores=True,
                                    output_hidden_states=True )
        
        last_hidden_states = outputs["hidden_states"][0][-1]
        logit = self.LLM.lm_head(last_hidden_states)[:, -1, :]
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        label = torch.LongTensor(label).to(self.args.device)
        batch_loss = loss_fct(logit, label )

        for index, output in enumerate(zip(outputs["sequences"])):
            pred = self.LLM_tokenizer.decode(output[0], skip_special_tokens=True)
            pred, hallucination_cnt = self.pasrse_record_res(my_input_list[index], label[index], pred[-self.args.max_new_tokens:], training_flag, record_flag)
            batch_pred.append(pred)
            batch_hallucination_cnt+=hallucination_cnt

        # batch_pred = []
        # batch_score = []
        # batch_hallucination_cnt = 0
        # dataset = Prompt_Dataset(my_input_list)
        # generator = self.pipe(dataset)
        # for index, generation in enumerate(generator):
        #     pred, hallucination_cnt = self.pasrse_record_res(my_input_list[index], label[index], generation[0]["generated_text"][-self.args.max_new_tokens:], training_flag, record_flag)
        #     batch_pred.append(pred)
        #     batch_score.append(generation[0]["scores"].squeeze())
        #     batch_hallucination_cnt+=hallucination_cnt

        return batch_pred, batch_hallucination_cnt, save_doc_num, batch_loss

    def pasrse_record_res(self, my_input, label, generation, training_flag, record_flag):
        pred, hallucination_cnt = extracted_label(generation)

        if training_flag:
            result_logger = self.train_result_logger
        else:    
            result_logger = self.test_result_logger

        if record_flag:
            result_logger.info(f"my_input: { my_input}")
            result_logger.info(f"generation: {generation}" )
            result_logger.info(f"label: {label}")
            result_logger.info(f"pred: {pred} "+ "\n========================================================================================================================")
        
        return pred, hallucination_cnt
    

   