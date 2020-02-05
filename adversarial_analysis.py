from collections import defaultdict
import json
def load_samples(l):
    res = []
    for file_path in l:
        with open(file_path, 'r') as f:
            items = [json.loads(s) for s in f]
        res.extend(items)
    return res

def flatten(answer, obj_type):
    res = []
    for temp in answer:
        if(type(temp)==list):
            res.extend([obj_type[k] for k in temp])
            # res.extend([str(k) for k in temp])
        else:
            res.append(temp)
    return  ' '.join(res)

def collect_statistics(samples):
    print('number of samples: ', len(samples))
    answer_ct = defaultdict(int)
    correct_ans_st = set()
    answer_rationale_dic = defaultdict()
    for sample in samples:
        answers = sample['answer_choices']
        obj_type = sample['objects']
        gold_ans_label = sample['answer_label']
        gold_answer = answers[gold_ans_label]
        gold_answer = flatten(gold_answer, obj_type)
        correct_ans_st.add(gold_answer)

        rationales = sample['rationale_choices']
        gold_rationale_label = sample['rationale_label']
        gold_rationale = rationales[gold_rationale_label]
        gold_rationale = flatten(gold_rationale, obj_type)

        # if(gold_answer in answer_rationale_dic):
        #     print(gold_answer)
        #     print(answer_rationale_dic[gold_answer])
        #     print()
        # assert(gold_answer not in answer_rationale_dic)
        answer_rationale_dic[gold_answer] = gold_rationale

    cor_ct = 0
    for sample in samples:
        obj_type = sample['objects']
        question =  flatten(sample['question'], obj_type)
        answers = sample['answer_choices']
        flag = True
        for answer in answers:
            ans_text = flatten(answer, obj_type)
            if(not ans_text in answer_rationale_dic):
                flag = False
                break
            # assert(ans_text in answer_rationale_dic), question+','+ans_text
        if(flag):
            cor_ct += 1
            for answer in answers:
                ans_text = flatten(answer, obj_type)
                print(question, ans_text, answer_rationale_dic[ans_text])


    # for sample in samples:
    #     answers = sample['answer_choices']
    #     obj_type = sample['objects']
    #     for answer in answers:
    #         ans_text = flatten(answer, obj_type)
    #         if(ans_text in correct_ans_st):
    #             answer_ct[ans_text]+=1
    # for key in answer_ct:
    #     print(key, answer_ct[key])
    print('fit samples(all answers has rationales): ', cor_ct)

def analyze_match_fold(samples, match_folder = 'val-0'):
    match_index = {x['match_index']: x for x in samples if x['match_fold'] == match_folder}
    print('match index length: ', len(match_index))
    example_index = 10
    example_sample = samples[example_index]

    answer_idx = 1
    source_sample = match_index[example_sample['answer_sources'][answer_idx]]
    print('test')


if __name__ == '__main__':
    train_json_file = '/mnt/ls15/scratch/users/yangshao/r2c_ori_data/train.jsonl'
    val_json_file = '/mnt/ls15/scratch/users/yangshao/r2c_ori_data/val.jsonl'
    # test_json_file = '/mnt/ls15/scratch/users/yangshao/r2c_ori_data/test.jsonl'
    # samples = load_samples([train_json_file, val_json_file])
    val_samples =  load_samples([val_json_file])
    print('finish loading validation samples!')

    analyze_match_fold(val_samples)

    # collect_statistics(samples)



