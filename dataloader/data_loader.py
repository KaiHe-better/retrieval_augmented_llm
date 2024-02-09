from dataloader.usmle_loader import get_loader_USMLE
from dataloader.ottqa_loader import get_loader_OTTQA
from dataloader.medmcqa_loader import get_loader_MedMCQA
from dataloader.mmlu_loader import get_loader_MMLU 
from dataloader.headqa_loader import get_loader_HEADQA 


def get_loader(args, triever_tokenizer):
    # print ('Loading data...')
    args.print_logger.info(f"Loading data ...")
    train_data_loader, dev_data_loader, test_data_loader = "", "", ""

    if args.dataset == "USMLE":
        train_file_path = "datasets/USMLE/questions/US/4_options/phrases_no_exclude_train.jsonl"
        dev_file_path = "datasets/USMLE/questions/US/4_options/phrases_no_exclude_dev.jsonl"
        test_file_path = "datasets/USMLE/questions/US/4_options/phrases_no_exclude_test.jsonl"

        rewrite_train_file_path = "datasets/USMLE/questions/US/4_options/rewrite_USMLE_train.json"

        rewrite_dev_file_path = "datasets/USMLE/questions/US/4_options/rewrite_USMLE_dev.json"
        # rewrite_dev_file_path = "datasets/USMLE/questions/US/4_options/rewrite_USMLE_test.json"

        rewrite_test_file_path = "datasets/USMLE/questions/US/4_options/rewrite_USMLE_test.json"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_USMLE(args, triever_tokenizer, train_file_path, dev_file_path, test_file_path, 
                                                                                      rewrite_train_file_path, rewrite_dev_file_path, rewrite_test_file_path
                                                                                      ) 

    elif args.dataset == "MedMCQA":
        train_file_path = "datasets/MedMCQA/train.json"
        dev_file_path = "datasets/MedMCQA/dev.json"

        # test_file_path = "datasets/MedMCQA/dev.json"
        test_file_path = "datasets/MedMCQA/test.json"

        rewrite_train_file_path = None
        rewrite_dev_file_path = "datasets/MedMCQA/rewrite_MedMCQA_dev.json"

        # rewrite_test_file_path = "datasets/MedMCQA/rewrite_MedMCQA_dev.json"
        rewrite_test_file_path = "datasets/MedMCQA/rewrite_MedMCQA_test.json"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_MedMCQA(args, triever_tokenizer, train_file_path, dev_file_path, test_file_path,
                                                                                        rewrite_train_file_path, rewrite_dev_file_path, rewrite_test_file_path,
                                                                                        ) 

    elif args.dataset == "OTTQA":
        train_file_path = "datasets/OTTQA/train.json"
        dev_file_path = "datasets/OTTQA/dev.json"
        test_file_path = "datasets/OTTQA/dev.json"
        # test_file_path = "datasets/OTTQA/test.json"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_OTTQA(args, triever_tokenizer, train_file_path, dev_file_path, test_file_path) 

    elif args.dataset == "MMLU":
        test_file_path = "datasets/MMLU/test.csv"
        rewrite_test_file_path = "datasets/MMLU/rewrite_MMLU_test.json"
        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_MMLU(args, triever_tokenizer, test_file_path, rewrite_test_file_path) 

    elif args.dataset == "HEADQA":
        train_file_path = "datasets/HEADQA/train.json"
        dev_file_path = "datasets/HEADQA/dev.json"
        test_file_path = "datasets/HEADQA/test.json"

        rewrite_train_file_path = "datasets/HEADQA/rewrite_HEADQA_train.json"
        rewrite_dev_file_path = "datasets/HEADQA/rewrite_HEADQA_test.json"
        rewrite_test_file_path = "datasets/HEADQA/rewrite_HEADQA_test.json"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_HEADQA(args, triever_tokenizer, train_file_path, dev_file_path, test_file_path,
                                                                                       rewrite_train_file_path, rewrite_dev_file_path, rewrite_test_file_path
                                                                                       ) 
    else:
        raise Exception("Wrong dataset selected !")


    
    return train_data_loader, dev_data_loader, test_data_loader