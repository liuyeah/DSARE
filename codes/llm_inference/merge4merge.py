import json
import numpy
import pdb


def cal_res(train_filepath, reference_test_filepath, success_filepath_list, fail_filepath, output_filepath):
    with open(train_filepath, 'r') as f_in:
        train = json.load(f_in)
    
    with open(reference_test_filepath, 'r') as f_in:
        reference_test = json.load(f_in)
        
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
    
    merged_dic = {}
    for success_filepath in success_filepath_list:
        with open(success_filepath, 'r') as f_in:
            for line in f_in:
                processed_line = json.loads(line)
                merged_dic[processed_line['id']] = processed_line
    
    with open(fail_filepath, 'r') as f_in:
        fail = json.load(f_in)
        for line in fail:
            line['pr'] = name
            merged_dic[line['id']] = line
            
    assert len(reference_test) == len(merged_dic)
    merged = []
    for line in reference_test:
        merged.append(
            merged_dic[line['id']]
        )
    
            
    with open(output_filepath, 'w') as f_out:
        json.dump(merged, f_out)


if __name__ == '__main__':
    
    cal_res(
        train_filepath = '../../datasets/example_data/train.json', 
        reference_test_filepath = '../../datasets/example_data/test.json',
        success_filepath_list = [
            './os.json',
        ], 
        fail_filepath = './no.json',
        output_filepath = './merged.json',
    )
    