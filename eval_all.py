"""
You can use this script to evaluate prediction files (valpreds.npy). Essentially this is needed if you want to, say,
combine answer and rationale predictions.
"""

import numpy as np
import json
import os
# get gt labels
# labels = {
#     'val_rationale': [],
#     'test_rationale': [],
#     'val_answer': [],
#     'test_answer': [],
# }
labels = {
    'val_rationale': [],
    # 'test_rationale': [],
    'val_answer': [],
    # 'test_answer': [],
}
root_folder = '/mnt/home/yangshao/vcr/action_data/'
for split in ('val',):
# for split in ('val', 'test'):
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
    # 'aug_rationale_flagship'
]

# root_folder = '/mnt/home/yangshao/vcr/r2c/models/saves/'
folder_to_preds = {folder: (np.load(os.path.join(root_folder, folder, 'valpreds.npy')),) for folder in folders}
# folder_to_preds = {folder: (np.load(os.path.join(root_folder, folder, 'valpreds.npy')),
#                             np.load(os.path.join(root_folder, folder, 'testpreds.npy'))) for folder in folders}

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
# print(folder_to_preds)
# for x, (y, z) in folder_to_preds.items():
#     assert np.abs(np.mean(y.sum(1)) - 1.0).sum() < 0.0001
#     assert np.abs(np.mean(z.sum(1)) - 1.0).sum() < 0.0001

# base_folders = sorted(set(['_'.join(x.split('_')[:-1]) for x in folder_to_preds]))
# print(base_folders)
answer_folder = 'aug_answer_flagship'
rationale_folder = 'aug_rationale_flagship'
# for folder in base_folders:
#     print("\n\n\nFor {}".format(folder), flush=True)
for split_id, split_name in enumerate(['val']):
# for split_id, split_name in enumerate(['val', 'test']):
    print("{}".format(split_name), flush=True)
    # Answer
    print(np.shape(folder_to_preds[answer_folder][split_id]))
    answer_hits = folder_to_preds[answer_folder][split_id].argmax(1) == labels[split_name + '_answer']
    # rationale_hits = folder_to_preds[rationale_folder][split_id].argmax(1) == labels[split_name + '_rationale']

    # joint_prob = []
    # for i in range(len(answer_hits)):
    #     a = folder_to_preds[answer_folder][split_id][i,:]
    #     b = folder_to_preds[rationale_folder][split_id][i,:]
    #     temp = np.outer(a, b)
    #     joint_prob.append(np.reshape(temp, 16))
    # joint_prob = np.array(joint_prob)
    # new_label = labels[split_name+'_answer']*4+labels[split_name+'_rationale']
    # joint_hits = joint_prob.argmax(1) == new_label
    print(" Answer acc: {:.3f}".format(np.mean(answer_hits)), flush=True)
    # print(" Rationale acc: {:.3f}".format(np.mean(rationale_hits)), flush=True)
    # print(" Joint acc: {:.3f}".format(np.mean(answer_hits & rationale_hits)), flush=True)
    # print(" Joint acc: {:.3f}".format(np.mean(joint_hits)), flush=True)
