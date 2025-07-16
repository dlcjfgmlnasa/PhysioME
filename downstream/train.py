# -*- coding:utf-8 -*-
import os
import copy
import sys
import torch
import random
import warnings
import numpy as np
sys.path.append(os.path.abspath('.'))

import argparse
from scipy.special import softmax
from typing import List
import torch.nn as nn
import torch.optim as opt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from dataset.utils import group_cross_validation
from downstream.data_loader import TorchDataset
from downstream.model import PhysioMEClassifier
from downstream.utils import load_pretrained_to_classifier
from models.utils import model_size

warnings.filterwarnings(action='ignore')

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Pretrained Checkpoint Hyperparameter
    parser.add_argument('--base_path', default=os.path.join('..', 'data', 'sleep_edfx'), type=str)
    parser.add_argument('--holdout_subject_size', default=30, type=int)
    parser.add_argument('--test_size', default=0.30, type=float)

    parser.add_argument('--pretrain_ckpt_path',
                        default=os.path.join('..', '..', 'ckpt', 'sleep_edfx', 'physiome'),
                        type=str)
    parser.add_argument('--class_num', default=5, type=int)

    # Modality 1 => ['EEG Fpz-Cz']
    # Modality 2 => ['EEG Fpz-Cz', 'EOG horizontal']
    # Modality 3 => ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
    parser.add_argument('--select_ch_names', default=['EEG Fpz-Cz'], type=List)
    parser.add_argument('--sfreq', default=100, type=int)

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=30, type=int)
    parser.add_argument('--train_base_learning_rate', default=0.0001, type=float)
    parser.add_argument('--train_batch_size', default=512, type=int)
    parser.add_argument('--train_batch_accumulation', default=1, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_paths, self.eval_paths = self.data_paths()
        self.model_ckpt_path = os.path.join(args.pretrain_ckpt_path, 'model', 'best_model.pth')
        self.model, self.model_param = load_pretrained_to_classifier(ckpt_path=self.model_ckpt_path,
                                                                     n_classes=args.class_num)
        self.model.to(device)
        self.ch_names = self.model.modal_names

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.clipping_norm_value = 2

        print('[Model Parameter]')
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Leaning Rate : {0}'.format(self.lr))

    def train(self):
        train_dataloader, eval_dataloader = self.load_dataloader()
        best_model_state, best_score, best_result = None, 0.0, {}
        for epoch in range(self.args.train_epochs):
            # 1. Train
            step = 0
            self.model.eval()
            self.model.fc.train()
            self.optimizer.zero_grad()

            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                x = {ch_name: x[:, i, :].squeeze() for i, ch_name in enumerate(self.ch_names)
                     if ch_name in self.args.select_ch_names}
                out = self.model(x)
                loss, _ = self.get_loss_and_performance(out, y)
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_norm_value)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                step += 1

            # 2. Test
            test_pred_prob, test_pred, test_real = [], [], []
            self.model.eval()
            self.model.fc.eval()
            for x, y in eval_dataloader:
                with torch.no_grad():
                    x, y = x.to(device), y.to(device)
                    x = {ch_name: x[:, i, :].squeeze() for i, ch_name in enumerate(self.ch_names)
                         if ch_name in self.args.select_ch_names}
                    out = self.model(x)
                    _, (pred, pred_prob, real) = self.get_loss_and_performance(out, y)
                    test_pred.extend(pred)
                    test_real.extend(real)
                    test_pred_prob.extend(pred_prob)

            # 3. Evaluation
            acc, auc = accuracy_score(y_true=test_real, y_pred=test_pred), \
                       roc_auc_score(y_true=test_real, y_score=test_pred_prob, multi_class='ovr')
            print('[Epoch] : {0:03d} \t [ACC] : {1:02.2f} \t [AUC] : {2:02.2f}'.format(epoch,
                                                                                       acc * 100,
                                                                                       auc * 100))
            if auc > best_score:
                best_score = auc
                best_model_state = self.model.state_dict()
                best_result = {'real': test_real, 'pred': test_pred, 'pred_prob': test_pred_prob}

        self.save_ckpt(model_state=best_model_state, result=best_result)

    def get_loss_and_performance(self, pred, real):
        loss = self.criterion(pred, real)
        pred_prob = copy.deepcopy(pred.detach().cpu().numpy())
        pred_prob = softmax(pred_prob, axis=-1)
        pred = list(torch.argmax(pred, dim=-1).detach().cpu().numpy())
        real = list(real.detach().cpu().numpy())
        return loss, (pred, pred_prob, real)

    def save_ckpt(self, model_state, result):
        if not os.path.exists(os.path.join(self.args.pretrain_ckpt_path, 'linear_prob',
                                           '{}'.format('.'.join(self.args.select_ch_names)))):
            os.makedirs(os.path.join(self.args.pretrain_ckpt_path, 'linear_prob',
                                     '{}'.format('.'.join(self.args.select_ch_names))))

        ckpt_path = os.path.join(self.args.pretrain_ckpt_path, 'linear_prob',
                                 '{}'.format('.'.join(self.args.select_ch_names)), 'best_model.pth')

        unimodal_param, multimodal_param = self.model_param
        torch.save({
            'model_nme': 'PhysioMEClassifier',
            'ch_names': self.args.select_ch_names,
            'backbone_networks_param': unimodal_param,
            'entire_networks_param': {
                'backbone_embed_dim': multimodal_param['backbone_embed_dim'],
                'num_backbone_frames': multimodal_param['backbone_num_frames'],
                'encoder_embed_dim': multimodal_param['encoder_embed_dim'],
                'encoder_heads': multimodal_param['encoder_heads'],
                'encoder_depths': multimodal_param['encoder_depths'],
                'decoder_embed_dim': multimodal_param['decoder_embed_dim'],
                'decoder_heads': multimodal_param['decoder_heads'],
                'decoder_recon_depths': multimodal_param['decoder_recon_depths'],
                'n_classes': self.args.class_num
            },
            'model_state': model_state,
            'result': result,
            'hyperparameter': self.args.__dict__,
        }, ckpt_path)

    def data_paths(self):
        paths = group_cross_validation(base_path=self.args.base_path,
                                       test_size=self.args.test_size,
                                       holdout_subject_size=self.args.holdout_subject_size)
        _, train_paths, eval_paths = paths['train_paths'], paths['val_paths'], paths['eval_paths']
        return train_paths, eval_paths

    def load_dataloader(self):
        # 1. Dataset
        train_dataset = TorchDataset(paths=self.train_paths, ch_names=self.ch_names, sfreq=self.args.sfreq)
        eval_dataset = TorchDataset(paths=self.eval_paths, ch_names=self.ch_names, sfreq=self.args.sfreq)

        # 2. DataLoader
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.eff_batch_size)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.eff_batch_size)
        return train_dataloader, eval_dataloader


if __name__ == '__main__':
    augments = get_args()
    Trainer(augments).train()
