# -*- coding:utf-8 -*-
import os
import sys
sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

import mne
import torch
import yaml
import random
import shutil
import argparse
import warnings
import numpy as np
import torch.optim as opt
from models.utils import model_size
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from dataset.utils import group_cross_validation
from pretrained.dp_neuronet.data_loader import TorchDataset
from pretrained.dp_neuronet.augmentation import DataAugmentationNeuroNet
from models.dp_neuronet.model import NeuroNet


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
    parser.add_argument('--config_yaml',
                        type=str,
                        default=os.path.join('..', '..', 'config', 'sleep_edfx', 'dp_neuronet.yaml'))
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
        return argparse.Namespace(**config_dict)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = NeuroNet(
            fs=args.rfreq, second=args.second, time_window=args.time_window, time_step=args.time_step,
            encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads,
            encoder_depths=args.encoder_depths,
            decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
            decoder_depths=args.decoder_depths,
            projection_hidden=args.projection_hidden, temperature=args.temperature
        ).to(device)

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.train_paths, self.val_paths, self.eval_paths = self.data_paths()
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.train_epochs)
        self.tensorboard_path = os.path.join(self.args.ckpt_path, self.args.model_name,
                                             self.args.ch_names[self.args.ch_idx], 'tensorboard')
        self.transform = DataAugmentationNeuroNet(prob=args.data_augmentation_prob)

        # remote tensorboard files
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        print('[NeuroNet Parameter]')
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Modal Name : {0}'.format(self.args.ch_names[self.args.ch_idx]))
        print('   >> Frame Size : {0}'.format(self.model.num_patches))
        print('   >> Leaning Rate : {0}\n'.format(self.lr))

    def train(self):
        train_dataset = TorchDataset(paths=self.train_paths, sfreq=self.args.sfreq, rfreq=self.args.rfreq,
                                     ch_idx=self.args.ch_idx,
                                     scaler=self.args.data_scaler)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        val_dataset = TorchDataset(paths=self.val_paths, sfreq=self.args.sfreq, rfreq=self.args.rfreq,
                                   ch_idx=self.args.ch_idx,
                                   scaler=self.args.data_scaler,
                                   downsampling=self.args.class_downsampling)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.train_batch_size, drop_last=True)
        eval_dataset = TorchDataset(paths=self.eval_paths, sfreq=self.args.sfreq, rfreq=self.args.rfreq,
                                    ch_idx=self.args.ch_idx,
                                    scaler=self.args.data_scaler)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.train_batch_size, drop_last=True)

        total_step = 0
        best_model_state, best_score = self.model.state_dict(), 0
        for epoch in range(self.args.train_epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            for x, _ in train_dataloader:
                x = x.to(device)
                x1, x2 = self.transform(x.unsqueeze(dim=1))
                x1, x2 = x1.squeeze(), x2.squeeze()
                out = self.model(x1, x2, mask_ratio=self.args.mask_ratio)
                recon_loss, contrastive_loss1, contrastive_loss2 = out
                loss = recon_loss + contrastive_loss1 + contrastive_loss2
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.args.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:08d}  '
                          '[Recon Loss] : {2:02.4f}  '
                          '[Contra Loss 1] : {3:02.4f}  '
                          '[Contra Loss 2] : {4:02.4f}  '
                          '[Total Loss] : {5:02.4f}'.format(
                            epoch, total_step + 1, recon_loss, contrastive_loss1, contrastive_loss2, loss))

                self.tensorboard_writer.add_scalar('Reconstruction Loss', recon_loss, total_step)
                self.tensorboard_writer.add_scalar('Contrastive Loss1', contrastive_loss1, total_step)
                self.tensorboard_writer.add_scalar('Contrastive Loss2', contrastive_loss2, total_step)
                self.tensorboard_writer.add_scalar('Total Loss', loss, total_step)

                step += 1
                total_step += 1

            val_acc, val_mf1 = self.linear_probing(val_dataloader, eval_dataloader)

            if val_acc > best_score:
                best_model_state = self.model.state_dict()
                best_score = val_acc

            print('[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n'.format(
                epoch, val_acc * 100, val_mf1 * 100))
            self.tensorboard_writer.add_scalar('Validation Accuracy', val_acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', val_mf1, total_step)

            self.optimizer.step()
            self.scheduler.step()

        self.save_ckpt(model_state=best_model_state)

    def linear_probing(self, val_dataloader, eval_dataloader):
        self.model.eval()
        (train_x, train_y), (test_x, test_y) = self.get_latent_vector(val_dataloader), \
                                               self.get_latent_vector(eval_dataloader)
        model = KNeighborsClassifier()
        model.fit(train_x, train_y)
        out = model.predict(test_x)
        acc, mf1 = accuracy_score(test_y, out), f1_score(test_y, out, average='macro')
        self.model.train()
        return acc, mf1

    def get_latent_vector(self, dataloader):
        total_x, total_y = [], []
        with torch.no_grad():
            for data in dataloader:
                x, y = data
                x, y = x.to(device), y.to(device)
                latent = self.model.forward_latent(x)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())
        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(total_y, axis=0)
        return total_x, total_y

    def save_ckpt(self, model_state):
        modal_name = self.args.ch_names[self.args.ch_idx]
        ckpt_path = os.path.join(self.args.ckpt_path, self.args.model_name, modal_name, 'model')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        torch.save({
            'model_name': 'DP-NeuroNet',
            'model_state': model_state,
            'model_parameter': {
                'fs': self.args.rfreq, 'second': self.args.second,
                'time_window': self.args.time_window, 'time_step': self.args.time_step,
                'encoder_embed_dim': self.args.encoder_embed_dim, 'encoder_heads': self.args.encoder_heads,
                'encoder_depths': self.args.encoder_depths,
                'decoder_embed_dim': self.args.decoder_embed_dim, 'decoder_heads': self.args.decoder_heads,
                'decoder_depths': self.args.decoder_depths,
                'projection_hidden': self.args.projection_hidden, 'temperature': self.args.temperature
            },
            'hyperparameter': self.args.__dict__,
            'paths': {'train_paths': self.train_paths, 'val_paths': self.val_paths, 'eval_paths': self.eval_paths}
        }, os.path.join(ckpt_path, 'best_model.pth'))

    def data_paths(self):
        paths = group_cross_validation(base_path=self.args.base_path,
                                       test_size=self.args.test_size,
                                       holdout_subject_size=self.args.holdout_subject_size)
        train_paths, val_paths, eval_paths = paths['train_paths'], paths['val_paths'], paths['eval_paths']
        return train_paths, val_paths, eval_paths

    @staticmethod
    def compute_metrics(output, target):
        output = output.argmax(dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(torch.float32))
        return accuracy


if __name__ == '__main__':
    augments = get_args()
    augments = load_config(path=augments.config_yaml)
    trainer = Trainer(augments)
    trainer.train()

