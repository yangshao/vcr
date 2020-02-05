import os
import h5py
import numpy as np
print(h5py.__version__)
print(os.getcwd())
# path = '/mnt/ls15/scratch/users/yangshao/r2c_ori_data/'
# path = '../train_val_data/'
# path = '../action_old_data/'
# split = 'train'
path = '../action_data/'
split = 'train'
# split = 'test'
# h5fn = os.path.join(path, f'bert_rationale_{split}.h5')
# bk_h5fn = os.path.join(path, f'bert_rationale_{split}_bk.h5')
# h5fn = os.path.join(path, f'beam_bert_rationale_{split}.h5')
h5fn = os.path.join(path, f'aug_bert_rationale_{split}.h5')
print(h5fn)
# print(bk_h5fn)
index = 0
with h5py.File(h5fn, 'r') as h5:
    print(len(h5))
    grp_items = {k: np.array(v, dtype=np.float16) for k,v in h5[str(index)].items()}
    print(len(grp_items.keys()))
    print(grp_items.keys())
# print(grp_items)
# with h5py.File(bk_h5fn, 'r') as h5:
#     print(len(h5))
#     bk_grp_items = {k: np.array(v, dtype=np.float16) for k,v in h5[str(index)].items()}
#     print(len(bk_grp_items.keys()))
#     print(bk_grp_items.keys())


