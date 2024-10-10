import os
import time
import json
import argparse

import torch

from constants import *
from utils import str2bool
from model import get_model_and_tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a dataset.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=1024,
        help='The maximum sequence length of the model.',
    )
    parser.add_argument(
        '--trust_remote_code',
        type=str2bool,
        default=False,
        help='Whether to trust the remote code.',
    )
    parser.add_argument(
        "--use_lora", 
        type=str2bool, 
        default=False, 
        help="Whether to use LoRA.")
    parser.add_argument(
        '--lora_dim',
        type=int,
        default=8,
        help='LoRA dimension.',
    )
    parser.add_argument(
        '--lora_scaling',
        type=int,
        default=32,
        help='LoRA scaling.',
    )
    parser.add_argument(
        "--lora_module_name", 
        type=str, 
        default="h.", 
        help="The scope of LoRA.")
    parser.add_argument(
        '--lora_load_path',
        type=str,
        default=None,
        help='The path to saved LoRA checkpoint.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    parser.add_argument(
        '--use_cuda',
        type=str2bool,
        default=True,
        help='Whether to use CUDA.',
    )
    parser.add_argument(
        '--output_dir_name',
        type=str,
        default='test',
        help='Where to store output files.',
    )
    args = parser.parse_args()
    return args


def get_instructions():
    # Instructions below follow https://huggingface.co/vicgalle/gpt2-alpaca.
    return [
        "Give three tips for a good meal.",
        "Write a poem about a delicious night.",
        "Write a tweet describing your capabilities.",
        "Pretend you are an alien visiting Earth. Write three opinions you believe."
    ]


def generate_response(model, tokenizer, instructions, device, output_dir):
    output_file = f"{output_dir}/response.txt"
    with open(output_file, "w") as f:
        for instruction in instructions:
            prompt = PROMPT_INPUT.format(input=instruction)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model.generate(**inputs, temperature=0.7, top_p=0.92, top_k=0, max_length=100, do_sample=True)
            f.write(tokenizer.decode(output[0], skip_special_tokens=True) + "\n\n")


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda"
    else:
        device = "cpu"
    lora_args = None
    if args.use_lora:
        lora_args = {"part_module_name": args.lora_module_name, "lora_dim": args.lora_dim, "lora_scaling": args.lora_scaling, "lora_load_path": args.lora_load_path}
    model, tokenizer = get_model_and_tokenizer(args.model_name_or_path, args.trust_remote_code, args.max_length, args.use_lora, lora_args)
    model.to(device)
    output_dir = os.path.join("./results", f"{args.output_dir_name}-{time.strftime('%Y%m%d-%H%M%S')}")
    assert not os.path.exists(output_dir), f"output directory {output_dir} already exists"
    os.makedirs(output_dir)
    # Save the args
    with open(os.path.join(output_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    instructions = get_instructions()
    generate_response(model, tokenizer, instructions, device, output_dir)


if __name__ == "__main__":
    main()
