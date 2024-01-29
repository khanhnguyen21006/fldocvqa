from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

from core.models import UdopConfig, UdopTokenizer, UdopUnimodelForConditionalGeneration

MODEL_CLASSES = {
    'UdopUnimodel': (UdopTokenizer, UdopConfig, UdopUnimodelForConditionalGeneration),
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    def __init__(self, args):
        self.model_name_or_path = getattr(args, 'model_name_or_path', None)
        self.model_type = getattr(args, 'model_type', None)
        self.config_name = getattr(args, "config_name", None)
        self.tokenizer_name = getattr(args, "tokenizer_name", None)
        self.resume_from = getattr(args, "resume_from", None)

        self.do_pretrain = not getattr(args, "docvqa", True)

        self.mae_version = getattr(args, "mae_version", 'mae_vit_base_patch16')
        self.mae_checkpoint = getattr(args, "mae_checkpoint", 'mae_ckpt/mae_pretrain_vit_base.pth')
        self.image_size = getattr(args, "image_size", 224)

        self.cache_dir = getattr(args, "cache_dir", None)
        self.model_revision = getattr(args, "model_revision", "main")
        self.use_auth_token = getattr(args, "use_auth_token", False)
        self.attention_type = getattr(args, "attention_type", "original_full")

def init_models(args, num_client, state_dict=None, return_meta=False):
    model_args = ModelArguments(args)

    if model_args.model_type in MODEL_CLASSES:
        tokenizer_type, config_type, model_type = MODEL_CLASSES[model_args.model_type]
    else:
        tokenizer_type, config_type, model_type =AutoTokenizer,  AutoConfig, AutoModelForTokenClassification
    tokenizer = tokenizer_type.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else (model_args.resume_from if model_args.resume_from else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    models = {_i: None for _i in range(num_client)}
    for _i in range(num_client):
        config = config_type.from_pretrained(
            model_args.config_name if model_args.config_name else (model_args.resume_from if model_args.resume_from else model_args.model_name_or_path),
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            attention_type=model_args.attention_type if model_args.attention_type else None,
            mae_version=model_args.mae_version,
            mae_checkpoint=model_args.mae_checkpoint,
            image_size=model_args.image_size,
        )
        models[_i] = model_type.from_pretrained(
            model_args.resume_from if model_args.resume_from else model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            state_dict=state_dict,
        )
        if model_args.do_pretrain:
            models[_i].resize_token_embeddings(len(tokenizer))

    if return_meta:
        model_metadata, layers = [], []
        for (k, v) in models[0].state_dict().items():
            model_metadata.append(v.shape)
            layer.append(k)
        return models if num_client > 1 else models[0], model_metadata, layers
    if num_client == 1:
        return models if num_client > 1 else models[0], tokenizer
    return models if num_client > 1 else models[0]

