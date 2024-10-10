import argparse
import torch
from trainer import Trainer
from utils import str2bool


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
        '--data_path',
        type=str,
        default='./data/alpaca_data.json',
        help='Path to the training data.',
        required=True,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Total number of training epochs to perform.',
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=8,
        help='Batch size for the training dataloader.',
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of backward steps to accumulate before performing an update.',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Initial learning rate (after the potential warmup period) to use.',
    )
    parser.add_argument(
        '--lr_warmup_ratio',
        type=float,
        default=0.0,
        help='Ratio of warm steps over total training steps for the lr scheduler.',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1.0e-6,
        help='Weight decay to use.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=16,
        help='Batch size for the eval dataloader.',
    )
    parser.add_argument(
        '--eval_ratio',
        type=float,
        default=0.01,
        help='Ratio of evalutaion split.',
    )
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=100,
        help='Interval of training steps to perform evaluation.',
    )
    parser.add_argument(
        '--output_dir_name',
        type=str,
        default='test',
        help='Where to store output files.',
    )
    args = parser.parse_args()
    return args


def train():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    train()
