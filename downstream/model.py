# -*- coding:utf -*-
import torch
import torch.nn as nn
from typing import Dict
from einops.layers.torch import Rearrange
from models.synthnet.model12 import PhysioME
from timm.models.vision_transformer import Block
from functools import partial


class PhysioMEClassifier(nn.Module):
    def __init__(self,
                 backbone_networks: Dict[str, nn.Module],
                 backbone_embed_dim: int, num_backbone_frames: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_recon_depths: int,
                 n_classes: int = 5):
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

        # [MultiModal Decoder - Restoration (for missing modality)]
        self.recon_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.multimodal_decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_embed_dim),
                                                         requires_grad=False)
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

        # [Classifier]
        hidden_embed_dim = encoder_embed_dim // 4
        self.fc = nn.Sequential(
            nn.Linear(encoder_embed_dim, hidden_embed_dim),
            nn.BatchNorm1d(hidden_embed_dim),
            nn.GELU(),
            nn.Linear(hidden_embed_dim, n_classes),
        )

    def forward(self, data):
        with torch.no_grad():
            x = self.inference_missing_modality(data=data)
        x = self.fc(x)
        return x

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
    from downstream.utils import load_pretrained_to_classifier
    ckpt_path = '/home/chlee/WorkSpace/MM/ckpt/sleep_edfx/synthnet/ablation/decoder_recon/128_4/model/best_model.pth'
    model_ = load_pretrained_to_classifier(
        ckpt_path=ckpt_path, n_classes=5
    )
    oo = model_({
        'EEG Fpz-Cz': torch.randn(64, 3000),
        'EEG Pz-Oz': torch.randn(64, 3000),
        'EOG horizontal': torch.randn(64, 3000)
    })
    print(oo.shape)
