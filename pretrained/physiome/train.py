# -*- coding:utf-8 -*-
import os
import sys
sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

import mne
import shutil
import torch
import yaml
import random
import warnings
import argparse
import numpy as np
from itertools import combinations
import torch.optim as opt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import OrderedDict
from models.utils import model_size
from dataset.utils import group_cross_validation
from torch.utils.tensorboard import SummaryWriter
from models.dp_neuronet.model import NeuroNet, NeuroNetEncoder
from models.physiome.model import PhysioME
from pretrained.physiome.data_loader import TorchDataset
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig


warnings.filterwarnings(action='ignore')

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml', type=str,
                        default=os.path.join('..', '..', 'config', 'sleep_edfx', 'physiome.yaml'))
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
        return argparse.Namespace(**config_dict)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_paths, self.val_paths, self.eval_paths = self.data_paths()
        self.ch_names = list(self.args.modality_name2path.keys())
        self.model = PhysioME(
            backbone_networks=self.get_encoder_backbone(),
            backbone_embed_dim=args.backbone_embed_dim, num_backbone_frames=args.backbone_num_frames,
            encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads,
            encoder_depths=args.encoder_depths,
            decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
            decoder_depths=args.decoder_depths,
            decoder_recon_depths=args.decoder_recon_depths,
            projection_hidden=args.projection_hidden, temperature=args.temperature,
        ).to(device)

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.train_epochs)
        self.tensorboard_path = os.path.join(self.args.ckpt_path, 'tensorboard')
        self.clipping_norm_value = 2.0

        # remote tensorboard files
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        print('[PhysioME Parameter]')
        print('   >> Modal Names : {0}'.format(', '.join(self.ch_names)))
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Leaning Rate : {0}'.format(self.lr))

    def train(self):
        train_dataset = TorchDataset(paths=self.train_paths, ch_names=self.ch_names,
                                     sfreq=self.args.sfreq, rfreq=self.args.rfreq, scaler=self.args.data_scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        val_dataset = TorchDataset(paths=self.val_paths, ch_names=self.ch_names,
                                   sfreq=self.args.sfreq, rfreq=self.args.rfreq, scaler=self.args.data_scaler,
                                   downsampling=self.args.class_downsampling)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.train_batch_size)
        eval_dataset = TorchDataset(paths=self.eval_paths, ch_names=self.ch_names,
                                    sfreq=self.args.sfreq, rfreq=self.args.rfreq, scaler=self.args.data_scaler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.train_batch_size)

        total_step = 0
        best_multimodal_model_state, best_score = self.model.state_dict(), 0
        for epoch in range(self.args.train_epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()
            for x, _ in train_dataloader:
                data = {
                    ch_name: torch.tensor(x[:, i, :].squeeze(), dtype=torch.float32).to(device)
                    for i, ch_name in enumerate(train_dataset.ch_names)
                }

                inter_recon_loss, missing_recon_loss, cross_contra_loss, cross_contra_acc = \
                    self.model(data=data, mask_ratio=self.args.mask_ratio)
                loss = inter_recon_loss + missing_recon_loss + cross_contra_loss
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_norm_value)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.args.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:07d}  '
                          '[Inter Recon Loss] : {2:2.4f}  '
                          '[Missing Recon Loss]: {3:2.4f}  '
                          '[Cross Contra Loss] : {4:2.4f}  '
                          '[Cross Contra Acc] : {5:2.3f}  '
                          '[Total Loss] : {6:2.3f}'.format(epoch, total_step + 1,
                                                           inter_recon_loss,
                                                           missing_recon_loss,
                                                           cross_contra_loss,
                                                           cross_contra_acc, loss))

                self.tensorboard_writer.add_scalar('Inter Recon Loss', inter_recon_loss, total_step)
                self.tensorboard_writer.add_scalar('Missing Recon Loss', missing_recon_loss, total_step)
                self.tensorboard_writer.add_scalar('Cross Contra Loss', cross_contra_loss, total_step)
                self.tensorboard_writer.add_scalar('Cross Contra Accuracy', cross_contra_acc, total_step)
                self.tensorboard_writer.add_scalar('Total Loss', loss, total_step)

                step += 1
                total_step += 1

            acc, mf1 = self.linear_probing(epoch, val_dataloader, eval_dataloader)

            self.tensorboard_writer.add_scalar('Validation Accuracy', acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', mf1, total_step)

            if mf1 > best_score:
                best_score = mf1
                best_multimodal_model_state = self.model.state_dict()
            self.scheduler.step()

        self.save_ckpt(best_multimodal_model_state)

    def linear_probing(self, epoch, val_dataloader, eval_dataloader):
        self.model.eval()
        modal_combinations = []
        for r in range(1, len(self.ch_names) + 1):
            modal_combinations.extend(combinations(self.ch_names, r))

        acc_list, mf1_list = [], []
        for modal_combination in modal_combinations:
            (train_x, train_y), (test_x, test_y) = self.get_latent_vector(modal_combination, val_dataloader), \
                                                   self.get_latent_vector(modal_combination, eval_dataloader)
            model = SVC()
            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)
            acc, mf1 = accuracy_score(test_y, pred_y), f1_score(test_y, pred_y, average='macro')
            acc_list.append(acc)
            mf1_list.append(mf1)
            print('[Epoch] : {0:03d} \t [{1}] => Accuracy : {2:2.4f} \t Macro-F1 : {3:2.4f}'.format(
                    epoch, ','.join(modal_combination), acc * 100, mf1 * 100))

        self.model.train()
        return np.mean(acc_list), np.mean(mf1_list)

    def get_latent_vector(self, modal_combination, dataloader):
        self.model.eval()
        total_x, total_y = [], []

        with torch.no_grad():
            for data in dataloader:
                x, y = data
                data = {
                    ch_name: x[:, i, :].squeeze().to(device)
                    for i, ch_name in enumerate(modal_combination)
                }
                latent = self.model.inference_missing_modality(data=data)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())

        total_x = np.concatenate(total_x, axis=0)
        total_y = np.concatenate(total_y, axis=0)

        self.model.train()
        return total_x, total_y

    def get_encoder_backbone(self):
        encoder_backbone = {}
        for ch_name, ckpt_path in self.args.modality_name2path.items():
            encoder_backbone[ch_name] = self.load_pretrained_unimodal(ckpt_path=ckpt_path)
        return encoder_backbone

    def data_paths(self):
        paths = group_cross_validation(base_path=self.args.base_path,
                                       test_size=self.args.test_size,
                                       holdout_subject_size=self.args.holdout_subject_size)
        train_paths, val_paths, eval_paths = paths['train_paths'], paths['val_paths'], paths['eval_paths']
        return train_paths, val_paths, eval_paths

    def load_pretrained_unimodal(self, ckpt_path):
        def get_find_parameter(model_state, find_name):
            param_dict = OrderedDict()
            for name, param in model_state.items():
                if name.find(find_name) != -1:
                    param_dict[name] = param
            return param_dict

        def change_parameter(pretrained_model_state, encoder_model, find_name):
            old_param = get_find_parameter(model_state=pretrained_model_state, find_name=find_name)
            new_param = encoder_model.state_dict()
            new_param = {on: nv for on, nv in zip(new_param.keys(), old_param.values())}
            return new_param

        # 1. pretrained model
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_parameter = ckpt['model_parameter']
        pretrained_model = NeuroNet(**model_parameter)
        pretrained_model.load_state_dict(ckpt['model_state'])

        # 2. NeuroNetEncoder
        backbone = NeuroNetEncoder(
            fs=model_parameter['fs'], second=model_parameter['second'],
            time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
            encoder_embed_dim=model_parameter['encoder_embed_dim'],
            encoder_heads=model_parameter['encoder_heads'],
            encoder_depths=model_parameter['encoder_depths']
        )
        backbone.frame_backbone.load_state_dict(change_parameter(
            pretrained_model_state=pretrained_model.state_dict(),
            encoder_model=backbone.frame_backbone,
            find_name='frame_backbone',
        ))
        backbone.patch_embed.load_state_dict(change_parameter(
            pretrained_model_state=pretrained_model.state_dict(),
            encoder_model=backbone.patch_embed,
            find_name='patch_embed',
        ))
        backbone.encoder_block.load_state_dict(change_parameter(
            pretrained_model_state=pretrained_model.state_dict(),
            encoder_model=backbone.encoder_block,
            find_name='encoder_block',
        ))
        backbone.encoder_norm.load_state_dict(change_parameter(
            pretrained_model_state=pretrained_model.state_dict(),
            encoder_model=backbone.encoder_norm,
            find_name='encoder_norm',
        ))
        backbone.cls_token = pretrained_model.autoencoder.cls_token
        backbone.pos_embed = pretrained_model.autoencoder.pos_embed

        # 3. Apply LoRA (Low-Rank Adaptation of Large Language Models)
        #  > https://arxiv.org/abs/2106.09685
        #  > https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
        peft_config = LoraConfig(
            r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=self.args.lora_dropout, bias='none',
            use_rslora=True, init_lora_weights='gaussian',
            target_modules=['attn.proj'],
        )
        backbone = get_peft_model(model=backbone, peft_config=peft_config)
        backbone.to(device)
        return backbone

    def save_ckpt(self, multimodal_model_state):
        ckpt_path = os.path.join(self.args.ckpt_path, 'model')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        backbone_ckpt = torch.load(list(self.args.modality_name2path.values())[0], map_location='cpu')
        backbone_parameter = backbone_ckpt['model_parameter']

        torch.save({
            'model_name': 'PhysioME',
            'ch_names': self.ch_names,
            'modality_backbone_param': {
                'fs': backbone_parameter['fs'], 'second': backbone_parameter['second'],
                'time_window': backbone_parameter['time_window'], 'time_step': backbone_parameter['time_step'],
                'encoder_embed_dim': backbone_parameter['encoder_embed_dim'],
                'encoder_heads': backbone_parameter['encoder_heads'],
                'encoder_depths': backbone_parameter['encoder_depths']
            },
            'entire_model_param': {
                'backbone_embed_dim': self.args.backbone_embed_dim,
                'backbone_num_frames': self.args.backbone_num_frames,
                'encoder_embed_dim': self.args.encoder_embed_dim,
                'encoder_heads': self.args.encoder_heads,
                'encoder_depths': self.args.encoder_depths,
                'decoder_embed_dim': self.args.decoder_embed_dim,
                'decoder_heads': self.args.decoder_heads,
                'decoder_depths': self.args.decoder_depths,
                'decoder_recon_depths': self.args.decoder_recon_depths,
                'projection_hidden': self.args.projection_hidden,
                'temperature': self.args.temperature
            },
            'model_state': multimodal_model_state,
            'hyperparameter': self.args.__dict__,
            'paths': {'train_paths': self.train_paths, 'val_paths': self.val_paths, 'eval_paths': self.eval_paths}
        }, os.path.join(ckpt_path, 'best_model.pth'))


if __name__ == '__main__':
    augments = get_args()
    augments = load_config(path=augments.config_yaml)
    trainer = Trainer(augments)
    trainer.train()
