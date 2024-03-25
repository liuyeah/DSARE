import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import json
import random
import time
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import copy
import pdb
from os import path as ospath
from transformers import AutoTokenizer, AutoModelForCausalLM


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


def f1_score(true, pred_result, rel2id):
    correct = 0
    total = len(true)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable']:
        if name in rel2id:
            neg = rel2id[name]
            break
    for i in range(total):
        golden = true[i]
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive +=1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    return result


def model_inference(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(0)
    generate_ids = model.generate(inputs.input_ids, do_sample=False, max_new_tokens=128)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    assert output[:len(prompt)] == prompt
    return output[len(prompt):]
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--api_key', '-ak', type=str, required=True)
    parser.add_argument('--train_path', '-tp', type=str, required=True, help="The path of training / demonstration data.")
    parser.add_argument('--test_path', '-ttp', type=str, required=True, help="The path of test data.")
    parser.add_argument('--output_success', '-os', type=str, required=True, help="The output directory of successful ICL samples.")
    parser.add_argument('--output_nores', '-on', type=str, required=True, help="The output directory of failed ICL samples.")
    parser.add_argument('--prompt', type=str, required=True, choices=["text", "text_schema", "instruct", "instruct_schema"])
    parser.add_argument('--k', type=int, default=1, help="k-shot demonstrations")
    parser.add_argument('--auto_modelpath', type=str, default="/data/liuyeah/projects/Relation_Extraction/hf-models/zephyr-7b-alpha", help="llama2 file path")
    
    args = parser.parse_args()
    
    # load automodel
    # max_memory={0: "22GiB", 1: "12GiB", "cpu":"80Gib"}
    tokenizer = AutoTokenizer.from_pretrained(args.auto_modelpath)
    model = AutoModelForCausalLM.from_pretrained(args.auto_modelpath, device_map="auto")
    
    print('successfully loaded the LLM: ', args.auto_modelpath)

    # Train / Demostration Set
    with open(args.train_path,'r') as f:
        train = json.load(f)
    label_list = {}
    for line in train:
        rel = line['relation']
        if rel not in label_list:
            label_list[rel] = [line]
        else:
            label_list[rel].append(line)


    # Relations
    rels = list(label_list.keys())
    rel2id = {}
    for i, rel in enumerate(rels):
        rel2id[rel] = i

    # Label words
    rel2labelword = {}
    for rel in rels:
        rel2labelword[rel] = rel.lower().replace("_"," ").replace("-", " ").replace("per", "person").replace("org", "organization").replace("stateor", "state or ")
    labelword2rel = {}
    for k,v in rel2labelword.items():
        labelword2rel[v] = k

    # Test Set
    with open(args.test_path,'r') as f:
        test = json.load(f)

    res = []
    true = []
    nores = []
    success = []
    with open(os.path.join(args.output_success, "os.json"),"w") as f:
        for input in tqdm(test):
            random.shuffle(rels)
            try:
                if "text" in args.prompt:
                    prompt = "There are candidate relations: " + ', '.join(labelword2rel.keys()) + ".\n"
                else:
                    prompt = "Given a context, a pair of head and tail entities in the context, decide the relationship between the head and tail entities from candidate relations: " + \
                     ', '.join(labelword2rel.keys()) + ".\n"
                
                knn_aug_data = input['knn_aug_data']
                kshot = knn_aug_data[:args.k]
                
                for data in kshot:
                    ss, se = data['subj_start'], data['subj_end']
                    head = ' '.join(data['token'][ss:se+1])
                    headtype = data['subj_type'].lower().replace('_',' ')
                    if headtype == "misc":
                        headtype = "miscellaneous"
                    os, oe = data['obj_start'], data['obj_end']
                    tail = ' '.join(data['token'][os:oe+1])
                    tailtype = data['obj_type'].lower().replace('_',' ')
                    if tailtype == "misc":
                        tailtype = "miscellaneous"
                    sentence = ' '.join([convert_token(token) for token in data['token']])
                    relation = rel2labelword[data['relation']]
                    if "schema" in args.prompt:
                        prompt += "Context: " + sentence + " The relation between " + headtype + " '" + head + "' and " + tailtype + " '" + tail + "' in the context is " + relation + ".\n"
                    else:
                        prompt += "Context: " + sentence + " The relation between '" + head + "' and '" + tail + "' in the context is " + relation + ".\n"
                
                        
                tss, tse = input['subj_start'], input['subj_end']
                testhead = ' '.join(input['token'][tss:tse+1])
                testheadtype = input['subj_type'].lower().replace('_',' ')
                if testheadtype == "misc":
                    testheadtype = "miscellaneous"
                tos, toe = input['obj_start'], input['obj_end']
                testtail = ' '.join(input['token'][tos:toe+1])
                testtailtype = input['obj_type'].lower().replace('_',' ')
                if testtailtype == "misc":
                    testtailtype = "miscellaneous"
                testsen = ' '.join(input['token'])
                if "schema" in args.prompt:
                    prompt += "Context: " + testsen + " The relation between " + testheadtype + " '" + testhead + "' and " + testtailtype + " '" + testtail + "' in the context is "
                else:
                    prompt += "Context: " + testsen + " The relation between '" + testhead + "' and '" + testtail + "' in the context is "
                    
                
                response = model_inference(model=model, prompt=prompt)
                resrel = response.strip().split('.')[0].lower()
                
                # pdb.set_trace()
                
                if resrel in labelword2rel:
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("city" in resrel) and (resrel.replace("city", "cities") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("city", "cities")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("city", "cities")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("country" in resrel) and (resrel.replace("country", "countries") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("country", "countries")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("country", "countries")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                elif ("province" in resrel) and (resrel.replace("province", "provinces") in labelword2rel):
                    truerel = rel2id[input['relation']]
                    predictrel = rel2id[labelword2rel[resrel.replace("province", "provinces")]]
                    true.append(truerel)
                    res.append(predictrel)
                    input['pr'] = labelword2rel[resrel.replace("province", "provinces")]
                    success.append(input)
                    f.writelines(json.dumps(input))
                    f.write('\n')
                else:
                    input['pr'] = resrel
                    nores.append(input)
            except Exception as e:
                print(e)
                # if e._message == 'You exceeded your current quota, please check your plan and billing details.':
                #     break
                nores.append(input)
                time.sleep(30)

    if len(nores)!=0:
        json.dump(nores, open(ospath.join(args.output_nores, "no.json"),'w'))
    # print(f1_score(true, res, rel2id))