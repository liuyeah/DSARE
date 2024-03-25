import pdb
import json


def str_list_compare(s_list1, s_list2):
    if len(s_list1) != len(s_list2):
        return False
    for idx in range(len(s_list1)):
        if s_list1[idx] != s_list2[idx]:
            return False
    return True


def process_token(text, textlower, headlower, taillower, hpos1, tpos1):
    token_origin = text.split(' ')
    token = textlower.split(' ')
    token_head = headlower.split(' ')
    token_tail = taillower.split(' ')

    if hpos1 == 0:
        ss = 0
    else:
        num = 0
        for idx in range(len(token)):
            word = token[idx]
            num += len(word) + 1
            if num == hpos1:
                ss = idx + 1
                break

    se = ss + len(token_head) - 1
        


    if tpos1 == 0:
        os = 0
    else:
        num = 0
        for idx in range(len(token)):
            word = token[idx]
            num += len(word) + 1
            if num == tpos1:
                os = idx + 1
                break

    oe = os + len(token_tail) - 1
    
    return token_origin, ss, se, os, oe


def process_line(DAdata):
    
    text = DAdata['text']
    truehead = DAdata['subj']
    hpos1, hpos2 = DAdata['subj_start'], DAdata['subj_end']
    truetail = DAdata['obj']
    tpos1, tpos2 = DAdata['obj_start'], DAdata['obj_end']
    relation = DAdata['relation']
    
    textlower = text.lower()
    headlower = truehead.lower()
    taillower = truetail.lower()
    
    try:
        token_origin, ss, se, os, oe = process_token(text, textlower, headlower, taillower, hpos1, tpos1)
    except:
        return False, {}
        
    
    processed_DAdata = {}
    processed_DAdata['text'] = text
    processed_DAdata['token'] = token_origin
    processed_DAdata['subj_start'], processed_DAdata['subj_end'], processed_DAdata['subj_type'] = ss, se, DAdata['subj_type']
    processed_DAdata['obj_start'], processed_DAdata['obj_end'], processed_DAdata['obj_type'] = os, oe, DAdata['obj_type']
    processed_DAdata['relation'] = relation
    processed_DAdata['ly_headlower'] = headlower
    processed_DAdata['ly_taillower'] = taillower
    
    return True, processed_DAdata



def merge(origin_filepath, da_filepath_list, output_filepath, add_k=8):
    with open(origin_filepath, 'r') as f_in:
        data = json.load(f_in)
    
    da_data = {}
    for line in data:
        if line['relation'] not in da_data:
            da_data[line['relation']] = []
    for da_filepath in da_filepath_list:
        with open(da_filepath, 'r') as f_in:
            for line in f_in:
                siganl, processed_line = process_line(json.loads(line))
                if siganl:
                    da_data[processed_line['relation']].append(processed_line)
    
    for relation in da_data:
        if len(da_data[relation]) < add_k:
            pdb.set_trace()
            # the number is not enough
        data.extend(da_data[relation][:add_k])
    
    with open(output_filepath, 'w') as f_out:
        json.dump(data, f_out, indent=True)


if __name__ == '__main__':
    merge(
        origin_filepath = '../../datasets/example_data/train.json',
        da_filepath_list = ['../../datasets/example_data/llm_generated.json',
                            ],
        output_filepath = '../../datasets/example_data/merged_train.json',
        # add_k indicates the num of augmented data per relation.
        add_k=8
    )