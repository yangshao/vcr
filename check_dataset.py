import os
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR, VCR_DIR
import json
from collections import defaultdict
def question_to_str(question_l):
    res = ''
    temp1 = [','.join([str(k) for k in temp]) if(type(temp)==list) else temp for temp in question_l]
    return ' '.join(temp1)
def load_dataset(file_name):
    dic = defaultdict(lambda: set())
    img_id_st = set()
    with open(os.path.join(VCR_ANNOTS_DIR, file_name), 'r') as f:
        items = [json.loads(s) for s in f]
    for i,item in enumerate(items):
        # img_id_st.add((item['img_id'],question_to_str(item['question'])))
        key  = item['img_id']+','+question_to_str(item['question'])
        img_id_st.add((item['img_id']+','+question_to_str(item['question'])))
        dic[key].add(i)

    # assert(len(items)==len(img_id_st)), str(len(items))+','+str(len(img_id_st))
    return items, img_id_st, dic

def check_common_ids(id_st1, id_st2):
    return id_st1.intersection(id_st2)

train_items, train_ids, train_dic = load_dataset('train.jsonl')
val_items, val_ids, val_dic = load_dataset('val.jsonl')
test_items,test_ids, test_dic = load_dataset('test.jsonl')
print('train ids: ', len(train_items), len(train_ids))
print('val ids: ', len(val_items),len(val_ids))
print('test ids: ', len(test_items), len(test_ids))

for key in train_dic:
    temp_st = train_dic[key]
    if(len(temp_st)>1):
        print('test')

train_val_common_id = check_common_ids(train_ids, val_ids)
print('train val common: ',len(train_val_common_id))
train_test_common_id = check_common_ids(train_ids, test_ids)
print('train test common: ',len(train_test_common_id))
