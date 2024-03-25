import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn, merge_collate_fn, loadLogger, find_best_performance
from prepro import TACREDProcessor, RETACREDProcessor
from evaluation import get_f1
from model import REModel
from torch.cuda.amp import GradScaler
import pdb


def evaluate(args, model, features, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    _, _, max_f1 = get_f1(keys, preds)

    output = {
        tag + "_f1": max_f1 * 100,
    }
    return max_f1, output
    
    

def merge_evaluate(args, model, features, id2label):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=merge_collate_fn, drop_last=False)
    keys, preds, gpt_prs, logits = [], [], [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  }
        keys += batch[2].tolist()
        gpt_prs += batch[5].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            logit = torch.softmax(logit, dim=-1)
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()
        # pdb.set_trace()
        logits += logit.tolist()

    # 到这里产生的是两个list，一个是gpt_prs，另一个是preds，这两个分别存储着传统模型和gpt输出的两种结果
    gpt_prs_labelname = []
    preds_labelname = []
    
    for idx in range(len(preds)):
        gpt_prs_labelname.append(id2label[gpt_prs[idx]])
        preds_labelname.append(id2label[preds[idx]])
        
    # pdb.set_trace()
    
    return gpt_prs_labelname, preds_labelname


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/tacred", type=str)
    parser.add_argument("--train_filename", default="train.json", type=str)
    parser.add_argument("--data_test_dir", default="./data/tacred", type=str)
    parser.add_argument("--data_merged_file", default="./data/tacred", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42)
    
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    
    parser.add_argument("--checkpoint_filepath", type=str, default="")
    
    parser.add_argument("--temp_output_merged_filepath", type=str, default="")

    args = parser.parse_args()
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    # config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    model = REModel(args, config)
    model.to(0)

    train_file = os.path.join(args.data_dir, args.train_filename)
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_test_dir, "test.json")

    # TACREDProcessor: processing TACRED and TACREV datasets
    processor = TACREDProcessor(args, tokenizer)
    # RETACREDProcessor: processing ReTACRED datasets
    # processor = RETACREDProcessor(args, tokenizer)
    train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)
    
    test_merged_features = processor.read(args.data_merged_file, merge_choice=True)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
    )
    
    model = torch.load(args.checkpoint_filepath)
    print('successfully load the model')
    # for tag, features in benchmarks:
    #     f1, output = evaluate(args, model, features, tag=tag)
    #     print(output)
    
    label2id = processor.LABEL_TO_ID
    id2label = {v: k for k, v in label2id.items()}
    
    
    gpt_prs_labelname, preds_labelname = merge_evaluate(args, model, test_merged_features, id2label)
    
    # generate the merged file
    output_merged = []
    
    with open(args.data_merged_file, 'r') as f_in:
        data_merged = json.load(f_in)
    
    assert len(data_merged) == len(gpt_prs_labelname)
    
    for idx in range(len(data_merged)):
        line = data_merged[idx]
        assert line['pr'] == gpt_prs_labelname[idx]
        line['gpt_pr'] = gpt_prs_labelname[idx]
        line['re_pr'] = preds_labelname[idx]
        output_merged.append(line)
    
    # pdb.set_trace()
    
    with open(args.temp_output_merged_filepath, 'w') as f_out:
        json.dump(output_merged, f_out)
        
    


if __name__ == "__main__":
    main()
