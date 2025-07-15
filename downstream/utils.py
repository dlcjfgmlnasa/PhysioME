# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from collections import OrderedDict
from downstream.model import PhysioMEClassifier
from models.synthnet.model12 import PhysioME
from models.neuronet.model import NeuroNetEncoder
from peft import get_peft_model, LoraConfig


def load_pretrained_to_classifier(ckpt_path: str, n_classes: int):
    # Load PhysioME w/o Decoder
    def get_find_parameter(model_state, find_name):
        param_dict = OrderedDict()
        for name_, param_ in model_state.items():
            if name_.find(find_name) != -1:
                param_dict[name_] = param_
        return param_dict

    def change_parameter(pretrained_model_state, encoder_model, find_name):
        old_param = get_find_parameter(model_state=pretrained_model_state, find_name=find_name)
        new_param = encoder_model.state_dict()
        new_param = {on: nv for on, nv in zip(new_param.keys(), old_param.values())}
        return new_param

    ckpt = torch.load(ckpt_path, map_location='cpu')
    ch_names = ckpt['ch_names']
    unimodal_parameter, multimodal_parameter = ckpt['modality_backbone_param'], ckpt['entire_model_param']
    multimodal_model_state = ckpt['model_state']

    # 1. Load Unimodal (= NeuroNet) Pretrained Model
    neuronet_pretrained_model = {}
    for ch_name in ch_names:
        neuronet = NeuroNetEncoder(**unimodal_parameter)
        peft_config = LoraConfig(
            r=ckpt['hyperparameter']['lora_r'], lora_alpha=ckpt['hyperparameter']['lora_alpha'],
            lora_dropout=ckpt['hyperparameter']['lora_dropout'], bias='none',
            use_rslora=True, init_lora_weights='gaussian',
            target_modules=['attn.proj'],
        )
        neuronet = get_peft_model(model=neuronet, peft_config=peft_config)
        neuronet_pretrained_model[ch_name] = neuronet

    # 2. Load PhysioME Classifier
    backbone = PhysioMEClassifier(
        backbone_networks=neuronet_pretrained_model,
        backbone_embed_dim=multimodal_parameter['backbone_embed_dim'],
        num_backbone_frames=multimodal_parameter['backbone_num_frames'],
        encoder_embed_dim=multimodal_parameter['encoder_embed_dim'],
        encoder_heads=multimodal_parameter['encoder_heads'],
        encoder_depths=multimodal_parameter['encoder_depths'],
        decoder_embed_dim=multimodal_parameter['decoder_embed_dim'],
        decoder_heads=multimodal_parameter['decoder_heads'],
        decoder_recon_depths=multimodal_parameter['decoder_recon_depths'],
        n_classes=n_classes
    )
    # [Backbone Network]
    backbone.backbone_networks.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.backbone_networks,
        find_name='backbone_networks'
    ))
    backbone.backbone_embedded.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.backbone_embedded,
        find_name='backbone_embedded'
    ))
    backbone.modal_token_dict.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.modal_token_dict,
        find_name='modal_token_dict'
    ))

    # [Multimodal Encoder]
    backbone.multimodal_encoder_block.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.multimodal_encoder_block,
        find_name='multimodal_encoder_block'
    ))
    backbone.multimodal_encoder_norm.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.multimodal_encoder_norm,
        find_name='multimodal_encoder_norm'
    ))

    # [Multimodal Decoder - Restoration (for Missing Modality)]
    backbone.recon_embed.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.recon_embed,
        find_name='recon_embed'
    ))
    backbone.multimodal_recon_block_dict.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.multimodal_recon_block_dict,
        find_name='multimodal_recon_block_dict'
    ))
    backbone.multimodal_recon_norm.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.multimodal_recon_norm,
        find_name='multimodal_recon_norm'
    ))
    backbone.multimodal_recon_pred_dict.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.multimodal_recon_pred_dict,
        find_name='multimodal_recon_pred_dict'
    ))
    backbone.multimodal_encoder_pos_embed = nn.Parameter(multimodal_model_state['multimodal_encoder_pos_embed'])
    backbone.multimodal_decoder_pos_embed = nn.Parameter(multimodal_model_state['multimodal_decoder_pos_embed'])
    backbone.mask_token = nn.Parameter(multimodal_model_state['mask_token'])
    return backbone, (unimodal_parameter, multimodal_parameter)
