import json
import os
from collections import defaultdict
img_folder = '/mnt/home/yangshao/vcr/vcr1/vcr1images'
data_folder = '/mnt/home/yangshao/vcr/action_data/'
train_file = 'aug_train.jsonl'
val_file = 'aug_val.jsonl'
test_file = 'aug_test.jsonl'
srl_train_file = 'aug_train_srl.jsonl'
srl_val_file = 'aug_val_srl.jsonl'
srl_test_file = 'aug_test_srl_jsonl'
train_file_action_analysis = 'aug_train_action_vos.jsonl'
val_file_action_analysis = 'aug_val_action_vos.jsonl'
test_file_action_analysis = 'aug_test_action_vos.jsonl'
OBJS, QUESTION, ANSWER, ANS_LABEL, RATIONALE, AUG_RATIONALE, \
RATIONALE_LABEL, PRED, META, IMG_FN, BOX, WIDTH, HEIGHT, NAME, AUG_RATIONALE_LABEL = ['objects', 'question',
                                                                 'answer_choices', 'answer_label', 'aug_rationales',
                                                                 'rationale_choices', 'rationale_label',
                                                                 'crf_pred_label', 'metadata_fn',
                                                                 'img_fn', 'boxes', 'width', 'height', 'names',
                                                                'aug_answer_rationale_indexes']
GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']

def load_jsonl(file_path):
    res = []
    with open(file_path, 'r') as infile:
        for line in infile:
            res.append(json.loads(line))
    return res
def replace_obj_name(answer, objects):
    res = []
    for e in answer:
        if(type(e)==list):
            temp_l = []
            for sub_e in e:
                temp_l.append(objects[sub_e])
            res.append(' and '.join(temp_l))
        else:
            res.append(e)
    return res
def get_data(data_folder, train_file):
    train_path = os.path.join(data_folder, train_file)
    train_data = load_jsonl(train_path)
    return train_data

def collect_persons(sample):
    obj_to_type = sample[OBJS]
    all_sents = []
    all_sents.append(sample[QUESTION])
    for sent in sample[ANSWER]:
        all_sents.append(sent)
    for sent in sample[RATIONALE]:
        all_sents.append(sent)
    cur_person_idx = 0
    person_dic = defaultdict()
    for sent in all_sents:
        for tok in sent:
            if isinstance(tok, list):
                temp_l = []
                for int_name in tok:
                    obj_type = obj_to_type[int_name]
                    if(obj_type=='person'):
                        if(int_name not in person_dic):
                            person_dic[int_name] = GENDER_NEUTRAL_NAMES[cur_person_idx]
                            cur_person_idx = (cur_person_idx+1)%len(GENDER_NEUTRAL_NAMES)
    return person_dic


def normalize_sentence(sent, sample, person_idx_dic):
    obj_to_type = sample[OBJS]
    # person_idx_dic = defaultdict()
    new_sent = []
    sent_wo_and = []
    for tok in sent:
        if isinstance(tok, list):
            temp_l = []
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                if(obj_type!='person'):
                    temp_l.append(obj_type)
                    sent_wo_and.append(obj_type)
                else:
                    temp_l.append(person_idx_dic[int_name])
                    sent_wo_and.append(person_idx_dic[int_name])
            new_sent.extend(' and '.join(temp_l).split())
        else:
            tok = tok.strip().split()
            new_sent.extend(tok)
            sent_wo_and.extend(tok)

    new2old_idx = [0 for i in range(len(new_sent))]
    i = 0
    j = 0
    # print(new_sent, sent_wo_and)
    while(j<len(new_sent)):
        if(new_sent[j]==sent_wo_and[i]):
            new2old_idx[j] = i
            i += 1
            j += 1
        elif(new_sent[j] == 'and'):
            new2old_idx[j] = -1
            j += 1
    return ' '.join(new_sent), new2old_idx


def write_to_file(samples, path):
    with open(path, 'w') as outfile:
        for sample in samples:
            outfile.write(json.dumps(sample)+'\n')


def write_lists(lists, path):
    with open(path, 'w') as outfile:
        for l in lists:
            outfile.write(' '.join([str(e) for e in l])+'\n')