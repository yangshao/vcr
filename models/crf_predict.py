import argparse
import os
import shutil
import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm
import config
import os
import pickle
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(torch.__version__)

# from dataloaders.vcr import VCR, VCRLoader
# from dataloaders.vcr_test import  VCR, VCRLoader
from dataloaders.vcr_crf import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models

#################################
#################################
######## Data loading stuff
#################################
#################################

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-vcr_data',
    dest='vcr_data',
    help='vcr data location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)
parser.add_argument(
    '-aug_flag',
    dest='aug_flag',
    action='store_true',
)

args = parser.parse_args()
config.VCR_ANNOTS_DIR = args.__dict__['vcr_data']
print('vcr annots dir', config.VCR_ANNOTS_DIR)

params = Params.from_file(args.params)
train, val, test = VCR.splits(embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                              only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True), aug_flag=args.aug_flag)
#NUM_GPUS = torch.cuda.device_count()
#NUM_CPUS = multiprocessing.cpu_count()
NUM_GPUS = 2
NUM_CPUS = 8
print('number gpus: ', NUM_GPUS)
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        td[k] = {k2: v.cuda(async=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
            async=True)
    return td
# num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2*NUM_GPUS)-1
num_workers = 3
batch_size = 12
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': batch_size // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
# loader_params = {'batch_size': 1 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
test_loader = VCRLoader.from_dataset(test, **loader_params)

ARGS_RESET_EVERY = 100
print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)
print(str(params['model']))
model = Model.from_params(vocab=train.vocab, params=params['model'])
print('*'*100)
for submodule in model.detector.backbone.modules():
    if isinstance(submodule, BatchNorm2d):
        submodule.track_running_stats = False
    for p in submodule.parameters():
        p.requires_grad = False

if config.double_flag:
    model.double()
model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()

optimizer = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad],
                                  params['trainer']['optimizer'])

lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None

if os.path.exists(args.folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=args.folder,
                                                           learning_rate_scheduler=scheduler)
else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)

print("STOPPING. now running on the test set", flush=True)
# train_temp_questions = np.load(os.path.join(args.folder, f'temp_question.npy'))
# train_temp_preds = np.load(os.path.join(args.folder, f'temp_preds.npy'))
# Load best
restore_best_checkpoint(model, args.folder)
model.eval()
model.training = False
# for eval_set, name in [(val_loader, 'val')]:
for eval_set, name in [(val_loader, 'val'), (test_loader, 'test')]:
    print(name)
    test_probs = []
    test_labels = []
    test_qa_logits = []
    test_qr_logits = []
    test_ar_logits = []
    test_adapative_weights = []
    test_qa_att = []
    test_qr_att = []
    test_ar_att = []
    test_inv_ar_att = []

    test_factor_weights = []
    test_qa_logits = []
    test_qr_logits = []
    test_ar_logits = []
    # val_test = []
    for b, (time_per_batch, batch) in enumerate(time_batch(eval_set)):
        with torch.no_grad():
            batch = _to_gpu(batch)
            output_dict = model(**batch)
            test_probs.append(output_dict['label_logits'].detach().cpu().numpy())
            test_labels.append(batch['label'].detach().cpu().numpy())
            # test_qa_logits.append(output_dict['qa_logits'].detach().cpu().numpy())
            # test_qr_logits.append(output_dict['qr_logits'].detach().cpu().numpy())
            # test_ar_logits.append(output_dict['ar_logits'].detach().cpu().numpy())
            # test_adapative_weights.append(output_dict['adaptive_weight'].detach().cpu().numpy())
            # test_qa_att.append(output_dict['qa_att'].detach().cpu().numpy())
            # test_qr_att.append(output_dict['qr_att'].detach().cpu().numpy())
            # test_ar_att.append(output_dict['ar_att'].detach().cpu().numpy())
            # test_inv_ar_att.append(output_dict['inv_ar_att'].detach().cpu().numpy())

            # test_factor_weights.append(output_dict['factor_weights'].detach().cpu().numpy())
            # test_qa_logits.append(output_dict['qa_logits'].detach().cpu().numpy())
            # test_qr_logits.append(output_dict['qr_logits'].detach().cpu().numpy())
            # test_ar_logits.append(output_dict['ar_logits'].detach().cpu().numpy())


    # val_test = np.concatenate(val_test, 0)
    test_labels = np.concatenate(test_labels, 0)
    test_probs = np.concatenate(test_probs, 0)
    # test_qa_logits = np.concatenate(test_qa_logits, 0)
    # test_qr_logits = np.concatenate(test_qr_logits, 0)
    # test_ar_logits = np.concatenate(test_ar_logits, 0)
    # test_adapative_weights = np.concatenate(test_adapative_weights, 0)
    # test_qa_att = np.concatenate(test_qa_att, 0)
    # test_qr_att = np.concatenate(test_qr_att, 0)
    # test_ar_att = np.concatenate(test_ar_att, 0)

    # test_factor_weights = np.concatenate(test_factor_weights, 0)
    # test_qa_logits = np.concatenate(test_qa_logits)
    # test_qr_logits = np.concatenate(test_qr_logits)
    # test_ar_logits = np.concatenate(test_ar_logits)

    acc = float(np.mean(test_labels == test_probs.argmax(1)))
    print("Final {} joint accuracy is {:.3f}".format(name, acc))

    action_test_probs = test_probs.reshape((-1, 4, 16))
    action_test_probs = np.max(action_test_probs, axis=-1)
    action_test_labels = (test_labels/16).astype(int)
    action_acc = float(np.mean(action_test_labels == action_test_probs.argmax(1)))
    print('Final {} action accuracy is {:.3f}'.format(name, action_acc))

    np.save(os.path.join(args.folder, f'{name}preds.npy'), test_probs)
    # np.save(os.path.join(args.folder, f'{name}qa_logits.npy'), test_qa_logits)
    # np.save(os.path.join(args.folder, f'{name}qr_logits.npy'), test_qr_logits)
    # np.save(os.path.join(args.folder, f'{name}ar_logits.npy'), test_ar_logits)
    # np.save(os.path.join(args.folder, f'{name}adaptive_weights.npy'), test_adapative_weights)
    # np.save(os.path.join(args.folder, f'{name}qa_att.npy'), test_qa_att)
    # np.save(os.path.join(args.folder, f'{name}qr_att.npy'), test_qr_att)
    # np.save(os.path.join(args.folder, f'{name}ar_att.npy'), test_ar_att)
    # with open(os.path.join(args.folder, f'{name}ar_att.pkl'), 'wb') as outfile:
    #     pickle.dump(test_ar_att, outfile)
    # with open(os.path.join(args.folder, f'{name}inv_ar_att.pkl'), 'wb') as outfile:
    #     pickle.dump(test_inv_ar_att, outfile)


    # np.save(os.path.join(args.folder, f'{name}factor_weights.npy'), test_factor_weights)
    # np.save(os.path.join(args.folder, f'{name}qa_logits.npy'), test_qa_logits)
    # np.save(os.path.join(args.folder, f'{name}qr_logits.npy'), test_qr_logits)
    # np.save(os.path.join(args.folder, f'{name}ar_logits.npy'), test_ar_logits)

