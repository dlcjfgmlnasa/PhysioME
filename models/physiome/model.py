# -*- coding:utf-8 -*-
import random
import torch
import torch.nn as nn
from typing import Dict, List
from timm.models.vision_transformer import Block
from models.utils import get_2d_sincos_pos_embed_flexible
from models.loss import NTXentLoss
from functools import partial
from einops.layers.torch import Rearrange


class PhysioME(nn.Module):
    def __init__(self,
                 backbone_networks: Dict[str, nn.Module],
                 backbone_embed_dim: int, num_backbone_frames: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 decoder_recon_depths: int,
                 projection_hidden: List[int], temperature: float):
        super().__init__()
        self.modal_names = list(backbone_networks.keys())
        self.backbone_networks = nn.ModuleDict(backbone_networks)
        self.num_backbone_frames = num_backbone_frames
        self.backbone_embed_dim = backbone_embed_dim
        self.encoder_embed_dim, self.decoder_embed_dim = encoder_embed_dim, decoder_embed_dim

        self.input_size = (self.num_backbone_frames, self.encoder_embed_dim)
        self.patch_size = (1, self.encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w
        self.mlp_ratio = 4.

        # [Backbone Network]
        self.backbone_embedded = nn.ModuleDict({
            modal_name: nn.Sequential(
                nn.Linear(backbone_embed_dim, encoder_embed_dim),
                Rearrange('b t e -> b e t'),
                nn.BatchNorm1d(encoder_embed_dim),
                nn.ELU(),
                Rearrange('b e t -> b t e'),
                nn.Linear(encoder_embed_dim, encoder_embed_dim)
            )
            for modal_name in self.modal_names
        })
        self.modal_token_dict = nn.ParameterDict({
            modal_name: nn.Parameter(torch.zeros(1, num_backbone_frames, encoder_embed_dim))
            for modal_name in self.modal_names
        })

        # [MultiModal Encoder]
        self.multimodal_encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_embed_dim),
                                                         requires_grad=False)
        self.multimodal_encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.multimodal_encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # [MultiModal Decoder]
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.multimodal_decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_embed_dim),
                                                         requires_grad=False)
        self.multimodal_decoder_block_dict = nn.ModuleDict({
            modal_name: nn.ModuleList([
                Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6))
                for _ in range(decoder_depths)
            ])
            for modal_name in self.modal_names
        })
        self.multimodal_decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.multimodal_decoder_pred_dict = nn.ModuleDict({
            modal_name: nn.Linear(decoder_embed_dim, backbone_embed_dim, bias=True)
            for modal_name in self.modal_names
        })

        # [MultiModal Decoder - Restoration (for missing modality)]
        self.recon_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.multimodal_recon_block_dict = nn.ModuleDict({
            modal_name: nn.ModuleList([
                Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6))
                for _ in range(decoder_recon_depths)
            ])
            for modal_name in self.modal_names
        })
        self.multimodal_recon_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.multimodal_recon_pred_dict = nn.ModuleDict({
            modal_name: nn.Linear(decoder_embed_dim, backbone_embed_dim, bias=True)
            for modal_name in self.modal_names
        })

        # [Contrastive Learning]
        self.backbone_projector_dict = nn.ModuleDict({
            modal_name: self.get_projection_layer([backbone_embed_dim] + projection_hidden)
            for modal_name in self.modal_names
        })
        self.fusion_projector = self.get_projection_layer([encoder_embed_dim] + projection_hidden)
        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.initialize_weights()

    @staticmethod
    def get_projection_layer(projection_hidden):
        projectors = []
        for i, (h1, h2) in enumerate(zip(projection_hidden[:-1], projection_hidden[1:])):
            projectors.append(nn.Linear(h1, h2))
            if i != len(projection_hidden) - 2:
                projectors.append(nn.BatchNorm1d(h2))
                projectors.append(nn.ELU())
        return nn.Sequential(*projectors)

    def initialize_weights(self):
        multimodal_encoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.multimodal_encoder_pos_embed.shape[-1],
                                                                        (self.grid_h, self.grid_w), cls_token=False)
        self.multimodal_encoder_pos_embed.data.copy_(torch.from_numpy(multimodal_encoder_pos_embed).float().unsqueeze(0))
        multimodal_decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.multimodal_decoder_pos_embed.shape[-1],
                                                                        (self.grid_h, self.grid_w), cls_token=False)
        self.multimodal_decoder_pos_embed.data.copy_(torch.from_numpy(multimodal_decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        for model_name, modal_token in self.modal_token_dict.items():
            self.modal_token_dict[model_name] = torch.nn.init.normal_(modal_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, data, mask_ratio: float = 0.8):
        total_x = []
        unimodal_token_dict = {}
        mask_dict = {}
        ids_restore_dict = {}

        for unimodal_name, unimodal_x in data.items():
            encoder_out = self.backbone_networks[unimodal_name](unimodal_x).detach().contiguous()
            encoder_emb = self.backbone_embedded[unimodal_name](encoder_out).detach().contiguous()

            x = encoder_emb[:, 1:, :] + self.modal_token_dict[unimodal_name] + self.multimodal_encoder_pos_embed

            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            unimodal_token_dict[unimodal_name] = encoder_out
            mask_dict[unimodal_name] = mask
            ids_restore_dict[unimodal_name] = ids_restore
            total_x.append(x)

        # concatenation vector tokens
        x = torch.cat(total_x, dim=1).contiguous()

        # apply Transformer blocks
        for block in self.multimodal_encoder_block:
            x = block(x)
        x = self.multimodal_encoder_norm(x)

        return (x, mask_dict, ids_restore_dict), unimodal_token_dict

    def forward_decoder(self, data, ids_restores):
        present_modals = list(ids_restores.keys())
        if not present_modals:
            return None

        split_size = data.shape[1] // len(present_modals) if present_modals else 0
        reconstructed_patches = []

        for i, modal_name in enumerate(present_modals):
            start, end = i * split_size, (i + 1) * split_size
            x = data[:, start:end, :].contiguous()
            x = self.decoder_embed(x).contiguous()

            ids_restore = ids_restores[modal_name]
            num_missing = ids_restore.shape[1] - x.shape[1]
            if num_missing > 0:
                mask_tokens = self.mask_token.repeat(x.shape[0], num_missing, 1)
                x = torch.cat([x, mask_tokens], dim=1).contiguous()

            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = x + self.multimodal_decoder_pos_embed

            for block in self.multimodal_decoder_block_dict[modal_name]:
                x = block(x)

            x = self.multimodal_decoder_norm(x)
            x = self.multimodal_decoder_pred_dict[modal_name](x)
            reconstructed_patches.append(x.detach().contiguous())

        x = torch.cat(reconstructed_patches, dim=1) if reconstructed_patches else None
        return x

    def forward_restoration_decoder(self, data, ids_restores):
        present_modals = list(ids_restores.keys())

        split_size = data.shape[1] // len(present_modals) if present_modals else 0
        reconstructed_patches = []

        for i, modal_name in enumerate(present_modals):
            start, end = i * split_size, (i + 1) * split_size
            x = data[:, start:end, :].contiguous()
            x = self.recon_embed(x).contiguous()

            ids_restore = ids_restores[modal_name]
            num_missing = ids_restore.shape[1] - x.shape[1]
            if num_missing > 0:
                mask_tokens = self.mask_token.repeat(x.shape[0], num_missing, 1)
                x = torch.cat([x, mask_tokens], dim=1).contiguous()

            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = x + self.multimodal_decoder_pos_embed

            for block in self.multimodal_recon_block_dict[modal_name]:
                x = block(x)

            x = self.multimodal_recon_norm(x)
            x = self.multimodal_recon_pred_dict[modal_name](x)
            reconstructed_patches.append(x.detach().contiguous())

        x = torch.cat(reconstructed_patches, dim=1) if reconstructed_patches else None
        return x

    def forward(self, data, mask_ratio: float = 0.8):
        # 1. Masked Prediction
        (fusion_tokens, masks, ids_restores), unimodal_token_dict = self.forward_encoder(data, mask_ratio=mask_ratio)
        real_tokens = torch.cat([unimodal_token[:, 1:, ] for unimodal_token in unimodal_token_dict.values()], dim=1)
        pred_tokens = self.forward_decoder(fusion_tokens, ids_restores)
        mask_tokens = torch.cat([mask for mask in masks.values()], dim=-1)
        reconstruction_loss1 = self.forward_mae_loss(real_tokens, pred_tokens, mask_tokens)

        # 2. Contrastive Learning
        cross_contrastive_loss, cross_contrastive_accuracy = [], []
        fusion_token = torch.mean(fusion_tokens, dim=1)
        o1 = self.fusion_projector(fusion_token)

        for unimodal_name, unimodal_tokens in unimodal_token_dict.items():
            unimodal_token = torch.mean(unimodal_tokens, dim=1)
            o2 = self.backbone_projector_dict[unimodal_name](unimodal_token)
            contra_loss, (labels, logits) = self.contrastive_loss(o1, o2)
            cross_contrastive_loss.append(contra_loss)
            cross_contrastive_accuracy.append(
                torch.mean(
                    torch.eq(torch.argmax(logits, dim=-1), labels).to(torch.float32)
                )
            )
        cross_contrastive_loss = torch.stack(cross_contrastive_loss, dim=-1)
        cross_contrastive_loss = torch.mean(cross_contrastive_loss, dim=-1)

        cross_contrastive_accuracy = torch.stack(cross_contrastive_accuracy, dim=-1)
        cross_contrastive_accuracy = torch.mean(cross_contrastive_accuracy, dim=-1)

        # 3. Drop Modality Prediction Task (Using Test & Inference)
        total_x = []
        ids_restore_dict = {}
        unimodal_token_dict = {}
        recon_masks = []
        with torch.no_grad():   # stop gradient: multimodal encoder
            present_modals = list(data.keys())
            num_missing = random.randint(1, len(present_modals) - 1)    # Drop Modality
            present_modals = random.sample(present_modals, len(present_modals) - num_missing)
            missing_modals = [m for m in self.modal_names if m not in present_modals]
            drop_data = {k: v for k, v in data.items() if k in present_modals}

            for unimodal_name in present_modals:
                x_raw = drop_data[unimodal_name]
                encoder_out = self.backbone_networks[unimodal_name](x_raw).detach().contiguous()
                encoder_emb = self.backbone_embedded[unimodal_name](encoder_out).detach().contiguous()

                x = encoder_emb[:, 1:, :] + self.modal_token_dict[unimodal_name] + self.multimodal_encoder_pos_embed
                x, mask, ids_restore = self.random_masking(x, mask_ratio=0.0)
                ids_restore_dict[unimodal_name] = ids_restore
                unimodal_token_dict[unimodal_name] = encoder_out
                total_x.append(x)
                recon_masks.append(mask)

            for unimodal_name in missing_modals:
                x_raw = data[unimodal_name]
                encoder_out = self.backbone_networks[unimodal_name](x_raw).detach().contiguous()
                unimodal_token_dict[unimodal_name] = encoder_out
                b, l, d = x.shape[0], self.num_backbone_frames, self.encoder_embed_dim
                mask = torch.ones((b, l), device=ids_restore.device)
                recon_masks.append(mask)

            x = torch.cat(total_x, dim=1).contiguous()
            for block in self.multimodal_encoder_block:
                x = block(x)
            x = self.multimodal_encoder_norm(x)

            ids_restore_all = {}
            for missing_modal in missing_modals:
                b, l, d = x.shape[0], self.num_backbone_frames, self.encoder_embed_dim
                ids_restore = torch.arange(l, device=x.device).unsqueeze(0).repeat(b, 1)
                ids_restore_all[missing_modal] = ids_restore
            ids_restore_dict.update(ids_restore_all)

        real_tokens = torch.cat([unimodal_token[:, 1:, ] for unimodal_token in unimodal_token_dict.values()], dim=1)
        recon_tokens = self.forward_restoration_decoder(x, ids_restore_dict)
        recon_masks = torch.cat(recon_masks, dim=-1)
        reconstruction_loss2 = self.forward_mae_loss(real_tokens, recon_tokens, recon_masks)
        return reconstruction_loss1, reconstruction_loss2, cross_contrastive_loss, cross_contrastive_accuracy

    def inference_missing_modality(self, data):
        # w/ restoration decoder
        expected_modals = self.modal_names
        present_modals = [modal for modal in data if modal in self.modal_names]
        missing_modals = [m for m in expected_modals if m not in present_modals]

        if len(missing_modals) == 0:
            (fusion_tokens, masks, ids_restores), _ = self.forward_encoder(data, mask_ratio=0.0)
            return torch.mean(fusion_tokens, dim=1)

        total_x = []
        ids_restore_dict = {}
        unimodal_token_dict = {}
        for unimodal_name in present_modals:
            x_raw = data[unimodal_name]
            encoder_out = self.backbone_networks[unimodal_name](x_raw).detach().contiguous()
            encoder_emb = self.backbone_embedded[unimodal_name](encoder_out).detach().contiguous()

            x = encoder_emb[:, 1:, :] + self.modal_token_dict[unimodal_name] + self.multimodal_encoder_pos_embed
            x, mask, ids_restore = self.random_masking(x, mask_ratio=0.0)

            ids_restore_dict[unimodal_name] = ids_restore
            unimodal_token_dict[unimodal_name] = encoder_out
            total_x.append(x)

        x = torch.cat(total_x, dim=1).contiguous()
        for block in self.multimodal_encoder_block:
            x = block(x)
        x = self.multimodal_encoder_norm(x)

        ids_restore_all = {}
        for missing_modal in missing_modals:
            b, l, d = x.shape[0], self.num_backbone_frames, self.encoder_embed_dim
            ids_restore = torch.arange(l, device=x.device).unsqueeze(0).repeat(b, 1)
            ids_restore_all[missing_modal] = ids_restore
        ids_restore_dict.update(ids_restore_all)

        recon = self.forward_restoration_decoder(x, ids_restore_dict)
        recon_chunk = recon.chunk(chunks=len(self.modal_names), dim=1)

        total_x = []
        for i, unimodal_name in enumerate(self.modal_names):
            if unimodal_name in present_modals:
                x_raw = data[unimodal_name]
                encoder_out = self.backbone_networks[unimodal_name](x_raw).detach().contiguous()
                encoder_emb = self.backbone_embedded[unimodal_name](encoder_out).detach().contiguous()
                x = encoder_emb[:, 1:, :] + self.modal_token_dict[unimodal_name] + self.multimodal_encoder_pos_embed
            else:
                encoder_out = recon_chunk[i].detach().contiguous()
                encoder_emb = self.backbone_embedded[unimodal_name](encoder_out).detach().contiguous()
                x = encoder_emb + self.modal_token_dict[unimodal_name] + self.multimodal_encoder_pos_embed
            total_x.append(x)

        x = torch.cat(total_x, dim=1).contiguous()
        for block in self.multimodal_encoder_block:
            x = block(x)
        x = self.multimodal_encoder_norm(x)
        x = torch.mean(x, dim=1)
        return x

    def inference(self, data):
        # w/o restoration decoder
        output = []
        batch_size = None

        for modal_name in self.modal_names:
            if modal_name in data and data[modal_name] is not None:
                x = data[modal_name]
                if batch_size is None:
                    batch_size = x.shape[0]
                elif x.shape[0] != batch_size:
                    raise ValueError(
                        f"Inconsistent batch size for modality {modal_name}: expected {batch_size}, got {x.shape[0]}")

                encoder_emb = self.backbone_networks[modal_name](x).detach().contiguous()
                encoder_emb = self.backbone_embedded[modal_name](encoder_emb).detach().contiguous()

                out = encoder_emb[:, 1:, :] + self.modal_token_dict[modal_name] + self.multimodal_encoder_pos_embed
                output.append(out)

        if not output:
            raise ValueError("No valid modalities provided in the input data")

        x = torch.cat(output, dim=1).contiguous()

        for block in self.multimodal_encoder_block:
            x = block(x)
        x = self.multimodal_encoder_norm(x)

        fusion_token = torch.mean(x, dim=1)
        return fusion_token

    @staticmethod
    def forward_mae_loss(real: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    @staticmethod
    def random_masking(x, mask_ratio):
        n, l, d = x.shape
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


if __name__ == '__main__':
    from models.neuronet.model import NeuroNetEncoder
    import torch.optim as optim

    net_ = PhysioME(
        backbone_networks={
            'EEG': NeuroNetEncoder(
                fs=100, second=30,
                time_window=4, time_step=1,
                encoder_embed_dim=256,
                encoder_heads=4,
                encoder_depths=4
            ),
            'EOG': NeuroNetEncoder(
                fs=100, second=30,
                time_window=4, time_step=1,
                encoder_embed_dim=256,
                encoder_heads=4,
                encoder_depths=4
            ),
            'EMG': NeuroNetEncoder(
                fs=100, second=30,
                time_window=4, time_step=1,
                encoder_embed_dim=256,
                encoder_heads=4,
                encoder_depths=4
            ),
        },
        backbone_embed_dim=256,
        num_backbone_frames=27,
        encoder_embed_dim=512, encoder_heads=4, encoder_depths=4,
        decoder_embed_dim=256, decoder_heads=4, decoder_depths=4,
        decoder_recon_depths=8,
        projection_hidden=[1024, 512], temperature=0.1
    ).to(device='cuda:0')

    data_ = {
        'EEG': torch.randn(16, 3000).to('cuda:0'),
        # 'EOG': torch.randn(16, 3000).to('cuda:0'),
        # 'EMG': torch.randn(16, 3000).to('cuda:0'),
    }
    oo = net_.inference_missing_modality(data_)
    print(oo.shape)

