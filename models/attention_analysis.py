import numpy as np
import json
import os
from collections import defaultdict
labels = {
    'val_rationale': [],
    'test_rationale': [],
    'val_answer': [],
    'test_answer': [],
}

root_folder = '/mnt/home/yangshao/vcr/r2c/models/saves/'
att = defaultdict()
def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
for split in ('val', 'test'):
    with open(f'/mnt/home/yangshao/vcr/new_data/{split}.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            labels[f'{split}_answer'].append(item['answer_label'])
            labels[f'{split}_rationale'].append(item['rationale_label'])

    qa_att_path = os.path.join(root_folder, f'crf_flagship/{split}_qa_att.npy')
    qr_att_path = os.path.join(root_folder, f'crf_flagship/{split}_qr_att.npy')
    att[split] = [np.load(qa_att_path), np.load(qr_att_path)]

    answer_labels = labels[f'{split}_answer']
    rationale_labels = labels[f'{split}_rationale']
    correct_qa_att = []
    for i in range(len(answer_labels)):
        l = answer_labels[i]
        correct_qa_att.append(att[split][0][i, l, :])
    correct_qr_att = []
    for i in range(len(rationale_labels)):
        l = rationale_labels[i]
        correct_qr_att.append(att[split][1][i, l, :])

    kl_div = []
    for i in range(len(answer_labels)):
        kl_div.append(kl(correct_qa_att[i], correct_qr_att[i]))

    print(split, np.mean(kl_div))

