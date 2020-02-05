import json
import os
VCR_ANNOTS_DIR = '/mnt/home/yangshao/vcr/train_val_data'
GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Peyton', 'Skyler', 'Frankie', 'Pat', 'Quinn']
def tokenize(tokenized_sent, obj_to_type):
    res = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                # new_ind = old_det_to_new_ind[int_name]
                new_ind = int_name
                text_to_use = GENDER_NEUTRAL_NAMES[
                    new_ind % len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                res.append(text_to_use)
        else:
            res.append(tok)
    # print(tokenized_sent, obj_to_type, res)
    return u' '.join(res).encode('utf-8').strip()

def collect_split(split):
    # target_file = os.path.join(VCR_ANNOTS_DIR, '{}_question_sents.txt'.format(split))
    target_file = VCR_ANNOTS_DIR+"/"+'{}_question_sents.txt'.format(split)
    print(target_file)
    out_f = open(target_file, 'w')
    with open(os.path.join(VCR_ANNOTS_DIR, '{}.jsonl'.format(split)), 'r') as f:
        items = [json.loads(s) for s in f]
        for item in items:
            question = tokenize(item['question'], item['objects'])
            out_f.write(str(question.decode('utf-8'))+'\n')

if __name__ == '__main__':
    # split = 'train'
    # split = 'val'
    split = 'test'
    collect_split(split)
