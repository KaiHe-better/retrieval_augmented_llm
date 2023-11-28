from dataloader.usmle_loader import get_loader_USMLE


def get_loader(args):
    # print ('Loading data...')
    args.print_logger.info(f"Loading data ...")
    train_data_loader, dev_data_loader, test_data_loader = "", "", ""

    if args.dataset == "USMLE":
        train_file_path = "datasets/USMLE/questions/US/4_options/phrases_no_exclude_train.jsonl"
        dev_file_path = "datasets/USMLE/questions/US/4_options/phrases_no_exclude_dev.jsonl"
        test_file_path = "datasets/USMLE/questions/US/4_options/phrases_no_exclude_test.jsonl"

        train_data_loader, dev_data_loader, test_data_loader, args = get_loader_USMLE(args,  train_file_path, dev_file_path, test_file_path) 

    elif args.dataset == "XXX":
        pass
    
    else:
        raise Exception("Wrong dataset selected !")


    
    return train_data_loader, dev_data_loader, test_data_loader