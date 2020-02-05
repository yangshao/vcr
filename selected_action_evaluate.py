from collections import defaultdict
import kb_utils as data_config
import os
import numpy as np
def load_actions(path):
    act_st = []
    with open(path, 'r') as infile:
        for line in infile:
            line = line.strip()
            act_st.append(line)
    return act_st
def load_action_set(path, selected_actions):
    data = data_config.load_jsonl(path)
    res = defaultdict(list)
    labels = []
    for i, sample in enumerate(data):
        answer_label = sample[data_config.ANS_LABEL]
        rationale_label = sample[data_config.AUG_RATIONALE_LABEL][answer_label]
        final_label = answer_label*16+rationale_label
        labels.append(final_label)
        vos =  sample['vos']
        cur_act_st = set()
        for vo in vos:
            if type(vo) == str:
                if vo in selected_actions:
                    cur_act_st.add(vo)
            else:
                temp = '_'.join(vo)
                if temp in selected_actions:
                    cur_act_st.add(temp)
        for act in cur_act_st:
            res[act].append(i)
    return res, labels

def evaluate(preds, action_dic,labels):
    labels = np.array(labels)
    for act in action_dic:
        print('action: ', act)
        mask = action_dic[act]
        total_ct = np.sum(mask)
        mask_labels = labels[mask]
        mask_preds = preds[mask]
        acc = float(np.mean(mask_labels == mask_preds.argmax(1)))
        print('accuracy: ',acc)

if __name__ == '__main__':
    evaluate_modes = ['val', 'test']
    selected_actions = load_actions(os.path.join(data_config.data_folder, 'top_actions.txt'))
    val_action_dic, val_labels = load_action_set(os.path.join(data_config.data_folder, 'aug_val_action_vos.jsonl'), selected_actions)
    # for act in selected_actions:
    #     print(act, len(val_action_dic[act]), np.sum(val_action_dic[act]))
    test_action_dic, test_labels = load_action_set(os.path.join(data_config.data_folder, 'aug_test_action_vos.jsonl'), selected_actions)
    # for act in selected_actions:
    #     print(act, len(test_action_dic[act]), np.sum(test_action_dic[act]))

    #load predictions
    val_path = os.path.join(data_config.data_folder, 'aug_crf_multi_modular2_flagship/valpreds.npy')
    val_preds = np.load(val_path)

    test_path = os.path.join(data_config.data_folder, 'aug_crf_multi_modular2_flagship/testpreds.npy')
    test_preds = np.load(test_path)

    print('mode: val')
    evaluate(val_preds, val_action_dic, val_labels)
    print('mode: test')
    evaluate(test_preds, test_action_dic, test_labels)



