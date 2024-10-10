from typing import Dict

import torch
import transformers

from constants import *
from lora import convert_linear_layer_to_lora, only_optimize_lora_parameters


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_model_and_tokenizer(model_name_or_path="gpt2", trust_remote_code=False, max_length=1024, use_lora=False, lora_args=None):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=max_length,
        padding_side="right"
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if use_lora:
        model = convert_linear_layer_to_lora(model=model, part_module_name=lora_args["part_module_name"], lora_dim=lora_args["lora_dim"], lora_scaling=lora_args["lora_scaling"])
        if lora_args["lora_load_path"] is not None:
            model.load_state_dict(torch.load(lora_args["lora_load_path"], map_location=model.device), strict=False)
        model = only_optimize_lora_parameters(model)

    return model, tokenizer