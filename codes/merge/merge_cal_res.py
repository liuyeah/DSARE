import json
import numpy
import pdb



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


def cal_res(train_filepath, success_filepath_list, fail_filepath):
    with open(train_filepath, 'r') as f_in:
        train = json.load(f_in)
        
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
        
    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable']:
        if name in rel2id:
            neg = rel2id[name]
            break
    
    reference = []
    prediction = []
    for success_filepath in success_filepath_list:
        with open(success_filepath, 'r') as f_in:
            for line in f_in:
                processed_line = json.loads(line)
                reference.append(rel2id[processed_line['relation']])
                prediction.append(rel2id[processed_line['final_pr']])
    
    with open(fail_filepath, 'r') as f_in:
        fail = json.load(f_in)
        for line in fail:
            truerel = rel2id[line['relation']]
            reference.append(truerel)
            prediction.append(neg)
    
    pdb.set_trace()
    print(f1_score(reference, prediction, rel2id))


if __name__ == '__main__':
    
    cal_res(
        train_filepath = '../../datasets/example_data/train.json', 
        success_filepath_list = [
            './os.json',
        ], 
        fail_filepath = './no.json'
    )
    