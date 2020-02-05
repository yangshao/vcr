"""
You can use this script to evaluate prediction files (valpreds.npy). Essentially this is needed if you want to, say,
combine answer and rationale predictions.
"""

import numpy as np
import json
import os
from collections import defaultdict
from sklearn.metrics import average_precision_score
def top_k_acc(pred, gold, topk = 3):
    best_n = np.argsort(pred, axis=1)[:,-topk:]
    suc = 0
    for i in range(pred.shape[0]):
        if(gold[i] in best_n[i,:]):
            suc += 1
    return float(suc)/pred.shape[0]

def eval_map(pred, gold):
    map_scores = []
    for i in range(pred.shape[0]):
        y_gold = np.zeros(pred[i].shape[0])
        y_gold[gold[i]] = 1
        score = average_precision_score(y_gold, pred[i])
        map_scores.append(score)
    return np.mean(map_scores)
labels = {
    'val_rationale': [],
    'val_answer': [],
    'val_rationale_expand': []
}
# for split in ('val', 'test'):
# root_folder = '/mnt/ls15/scratch/users/yangshao/r2c_ori_data/'
root_folder = '/mnt/home/yangshao/vcr/action_data/'
items = []
for split in ('val',):
    with open(root_folder+f'aug_{split}.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            items.append(item)
            labels[f'{split}_answer'].append(item['answer_label'])
            # labels[f'{split}_rationale'].append(item['rationale_label'])
            labels[f'{split}_rationale'].append(item['aug_answer_rationale_indexes'][item['answer_label']])
            labels[f'{split}_rationale_expand'].append(item['aug_answer_rationale_indexes'])

for k in labels:
    labels[k] = np.array(labels[k])

folders = [
    # 'aug_crf_flagship',
    # 'aug_crf_multi_flagship',
    'aug_crf_multi_adaptive_flagship',
]

folder_to_preds = defaultdict(lambda: defaultdict())
for folder in folders:
    if( 'rationale' in folder):
        key1 = 'beamvalpreds.npy'
        # key2 = 'beamtestpreds.npy'
    else:
        key1 = 'valpreds.npy'
        # key2 = 'testpreds.npy'
    path1 = os.path.join(root_folder, folder, key1)
    # path2 = os.path.join(root_folder, folder, key2)
    # print(path1, path2)
    # folder_to_preds[folder] = (np.load(path1), np.load(path2))
    folder_to_preds[folder] = (np.load(path1),)

def _softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(1)[:,None]

# target_folder = 'aug_crf_flagship'
target_folder = 'aug_crf_multi_adaptive_flagship'
a_ct = 4
r_ct = 16


def get_error_statistics(split_name, joint_prob, labels):
    n = len(joint_prob)
    total_wrong_ct = 0
    w1, w2, w3, w4 = 0, 0, 0, 0
    for i in range(n):
        cur_joint_prob = joint_prob[i]
        cur_new_label = labels[split_name+'_answer'][i]*16+labels[split_name+'_rationale'][i]
        cur_answer_label = labels[split_name+'_answer'][i];
        cur_rational_label = labels[split_name+'_rationale'][i]
        pred_label = cur_joint_prob.argmax()
        if(pred_label == cur_new_label):
            continue
        total_wrong_ct += 1
        pred_answer = (int)(pred_label/16)
        pred_rationale = pred_label%16
        gold_pred_rationale = labels[split_name+'_rationale_expand'][i][pred_answer]
        if pred_answer == cur_answer_label and pred_rationale != cur_rational_label:
            w1 += 1
        elif pred_answer != cur_answer_label and pred_rationale == cur_rational_label:
            w2 += 1
        elif pred_answer != cur_answer_label and pred_rationale != cur_rational_label and pred_rationale == gold_pred_rationale:
            w3 += 1
        elif pred_answer != cur_answer_label and pred_rationale != cur_rational_label and pred_rationale != gold_pred_rationale:
            w4 += 1
    print(w1, w2, w3, w4, total_wrong_ct)
    print(1.0*w1/total_wrong_ct, 1.0*w2/total_wrong_ct, 1.0*w3/total_wrong_ct, 1.0*w4/total_wrong_ct)





def generate_predictions():
    for split_id, split_name in enumerate(['val']):
        print("{}".format(split_name), flush=True)
        # Answer
        n, ct = np.shape(folder_to_preds[target_folder][split_id])
        joint_prob = folder_to_preds[target_folder][split_id]


        new_label = labels[split_name+'_answer']*16+labels[split_name+'_rationale']
        joint_hits = joint_prob.argmax(1) == new_label
        joint_beam_top3 = top_k_acc(joint_prob, new_label, topk=3)
        joint_beam_map = eval_map(joint_prob, new_label)
        res = get_error_statistics(split_name, joint_prob, labels)
        print(" CRF acc: {:.3f}, top 3 acc: {:.3f}, map: {:.3f}".format(np.mean(joint_hits), joint_beam_top3, joint_beam_map), flush=True)


import copy
def generate_data(folder):
    cor_ct = 0
    for split_id, split_name in enumerate(['val']):
        print("{}".format(split_name), flush=True)
        print(folder)
        qa_logits = np.load(os.path.join(folder, split_name+'qa_logits.npy'))
        qr_logits = np.load(os.path.join(folder, split_name+'qr_logits.npy'))
        ar_logits = np.load(os.path.join(folder, split_name+'ar_logits.npy'))
        with open(os.path.join(folder, split_name+'_pred.jsonl'), 'w') as outfile:
            n, ct = np.shape(folder_to_preds[target_folder][split_id])
            joint_prob = folder_to_preds[target_folder][split_id]
            new_label = labels[split_name+'_answer']*16+labels[split_name+'_rationale']
            for i in range(len(new_label)):
                pred_label = joint_prob[i].argmax()
                gd_label = new_label[i]
                new_item = copy.deepcopy(items[i])
                new_item['crf_pred_label'] = str(pred_label)
                new_item['qa_logits'] = qa_logits[i].tolist()
                new_item['qr_logits'] = qr_logits[i].tolist()
                new_item['ar_logits'] = ar_logits[i].tolist()
                json.dump(new_item, outfile)
                outfile.write('\n')
def generate_error_data(folder):
    cor_ct = 0
    for split_id, split_name in enumerate(['val']):
        print("{}".format(split_name), flush=True)
        print(folder)
        qa_logits = np.load(os.path.join(folder, split_name+'qa_logits.npy'))
        qr_logits = np.load(os.path.join(folder, split_name+'qr_logits.npy'))
        ar_logits = np.load(os.path.join(folder, split_name+'ar_logits.npy'))
        with open(os.path.join(folder, split_name+'_errors.jsonl'), 'w') as outfile:
            n, ct = np.shape(folder_to_preds[target_folder][split_id])
            joint_prob = folder_to_preds[target_folder][split_id]
            new_label = labels[split_name+'_answer']*16+labels[split_name+'_rationale']
            for i in range(len(new_label)):
                pred_label = joint_prob[i].argmax()
                gd_label = new_label[i]
                if(pred_label != gd_label):
                    new_item = copy.deepcopy(items[i])
                    new_item['crf_pred_label'] = str(pred_label)
                    new_item['qa_logits'] = qa_logits[i].tolist()
                    new_item['qr_logits'] = qr_logits[i].tolist()
                    new_item['ar_logits'] = ar_logits[i].tolist()
                    json.dump(new_item, outfile)
                    outfile.write('\n')
                else:
                    cor_ct += 1;
            print(n,cor_ct, 1.0*cor_ct/n);





if __name__ == '__main__':
    # generate_error_data(root_folder+target_folder)
    # generate_data(root_folder+target_folder)
    generate_predictions()




