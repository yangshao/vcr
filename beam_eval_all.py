"""
You can use this script to evaluate prediction files (valpreds.npy). Essentially this is needed if you want to, say,
combine answer and rationale predictions.
"""

import numpy as np
import json
import os
from collections import defaultdict
from sklearn.metrics import average_precision_score
# get gt labels
# labels = {
#     'val_rationale': [],
#     'val_answer': [],
#     'test_rationale': [],
#     'test_answer': [],
# }
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
}
# for split in ('val', 'test'):
# root_folder = '/mnt/ls15/scratch/users/yangshao/r2c_ori_data/'
root_folder = '/mnt/home/yangshao/vcr/action_data/'
for split in ('val',):
    with open(root_folder+f'aug_{split}.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            labels[f'{split}_answer'].append(item['answer_label'])
            # labels[f'{split}_rationale'].append(item['rationale_label'])
            labels[f'{split}_rationale'].append(item['aug_answer_rationale_indexes'][item['answer_label']])
for k in labels:
    labels[k] = np.array(labels[k])

# folders = [
#     'flagship_answer',
#     'no_reasoning_rationale',
#     'no_reasoning_answer',
#     'flagship_rationale',
#     'no_question_answer',
#     'no_vision_answer',
#     'no_vision_rationale',
#     'bottom_up_top_down_glove_rationale',
#     'bottom_up_top_down_rationale',
#     'bottom_up_top_down_glove_answer',
#     'no_question_rationale',
#     'vqa_baseline_glove_answer',
#     'vqa_baseline_glove_rationale',
#     'bottom_up_top_down_answer',
#     'glove_answer',
#     'glove_rationale',
#     'mutan_glove_answer',
#     'mutan_glove_rationale',
#     'mlb_glove_answer',
#     'mlb_glove_rationale',
#     'esim_answer',
#     'esim_rationale',
#     'lstm_answer',
#     'lstm_rationale',
# ]
folders = [
    'aug_answer_flagship',
    'aug_rationale_flagship'
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



# folder_to_preds = {folder: (np.load(os.path.join(root_folder, folder, 'beamvalpreds.npy' if 'rationale' in folder else 'valpreds.npy')),
#                             np.load(os.path.join(root_folder, folder, 'beamtestpreds.npy' if 'rationale' in folder else 'testpreds.npy'))) for folder in folders}

def _softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(1)[:,None]

# log_folders = [
#     'bert_answer',
#     'bert_rationale',
#     'bert_answer_ending',
#     'bert_rationale_ending',
# ]
# for folder in log_folders:
#     new_name = {'bert_answer_ending': 'bert_ending_answer', 'bert_rationale_ending': 'bert_ending_rationale'}.get(folder, folder)
#     folder_to_preds[new_name] = (np.exp(np.load(os.path.join('/home/rowan/datasets3/vswagmodels/', folder, 'val-logprobs.npy'))),
#                                np.exp(np.load(os.path.join('/home/rowan/datasets3/vswagmodels/', folder, 'test-logprobs.npy'))))

# sanity check
# for x, (y, z) in folder_to_preds.items():
#     assert np.abs(np.mean(y.sum(1)) - 1.0).sum() < 0.0001
#     assert np.abs(np.mean(z.sum(1)) - 1.0).sum() < 0.0001

# base_folders = sorted(set(['_'.join(x.split('_')[:-1]) for x in folder_to_preds]))
# print(base_folders)
# base_folders=['flagship_answer_small', 'flagship_rationale_small']
answer_folder = 'aug_answer_flagship'
rationale_folder = 'aug_rationale_flagship'
# for folder in base_folders:
#     print("\n\n\nFor {}".format(folder), flush=True)
# for split_id, split_name in enumerate(['val', 'test']):
a_ct = 4
r_ct = 16

for split_id, split_name in enumerate(['val']):
    print("{}".format(split_name), flush=True)
    # Answer
    a_n, a_ct = np.shape(folder_to_preds[answer_folder][split_id])
    r_n, r_ct = np.shape(folder_to_preds[rationale_folder][split_id])
    reshaped_temp = np.reshape(folder_to_preds[rationale_folder][split_id], (a_n, a_ct, r_ct))

    answer_hits = folder_to_preds[answer_folder][split_id].argmax(1) == labels[split_name + '_answer']
    answer_label = labels[split_name+'_answer']
    gold_rationale_pred = reshaped_temp[np.arange(0, a_n), answer_label]
    rationale_hits = gold_rationale_pred.argmax(1) == labels[split_name + '_rationale']

    joint_prob = np.zeros((np.shape(reshaped_temp)))
    for i in range(a_n):
        for j in range(4):
            a = folder_to_preds[answer_folder][split_id][i][j]
            b = reshaped_temp[i][j]
            joint_prob[i][j] = a*b
    joint_prob = np.reshape(joint_prob, (a_n, a_ct*r_ct))

    # joint_prob = []
    # for i in range(len(answer_hits)):
    #     a = folder_to_preds[answer_folder][split_id][i,:]
    #     b = folder_to_preds[rationale_folder][split_id][i,:]
    #     temp = np.outer(a, b)
    #     joint_prob.append(np.reshape(temp, 16))
    # joint_prob = np.array(joint_prob)
    # new_label = labels[split_name+'_answer']*4+labels[split_name+'_rationale']
    new_label = labels[split_name+'_answer']*16+labels[split_name+'_rationale']
    joint_hits = joint_prob.argmax(1) == new_label
    joint_beam_top3 = top_k_acc(joint_prob, new_label, topk=3)
    joint_beam_map = eval_map(joint_prob, new_label)
    print(" Answer acc: {:.3f}".format(np.mean(answer_hits)), flush=True)
    print(" Rationale acc: {:.3f}".format(np.mean(rationale_hits)), flush=True)
    print(" Joint beam acc: {:.3f}, top 3 acc: {:.3f}, map: {:.3f}".format(np.mean(joint_hits), joint_beam_top3, joint_beam_map), flush=True)
    print(" Joint greedy acc: {:.3f}".format(np.mean(answer_hits & rationale_hits)), flush=True)



