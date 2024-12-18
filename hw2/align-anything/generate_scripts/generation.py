from __future__ import annotations
import json
import os
import argparse

import sys
import time

from tqdm.auto import tqdm
from vllm import SamplingParams, LLM



PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {prompt} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with gpt4',
    )
    # Model
    parser.add_argument(
       '--model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
       required=True,
    )
    parser.add_argument(
        '--output_name',
        type=str,
        help='the name of the output json file',
        required=True,

    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the eval output.',
    )
    parser.add_argument(
        '--input_path',
        type = str,
        default='problem.json',
        help=' Input file (Json) name'
    )
    parser.add_argument(
        '--num_responses',
        type = int,
        default=1,
        help='number of responses'
    )
    return parser.parse_args()


def generate_answer_by_vllim(problems: list[str], model_name_or_path: str, num_responses: int) ->list[str]:
    samplingparams = SamplingParams(
        temperature = 0.5,
        repetition_penalty = 1.1,
        max_tokens = 512,
        n=num_responses,
    )
    llm = LLM(
        model = model_name_or_path,
        tokenizer = model_name_or_path,
        tokenizer_mode='auto',
        trust_remote_code=False,
        download_dir=None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        swap_space=2,
        max_num_seqs=64,
    )
    prompts=[]
    for entry in problems:
        prompt = PROMPT_INPUT.format(prompt=entry['prompt'])
        prompts.append(prompt)

    outputs = llm.generate(prompts, samplingparams)

    answers = []
    for output, entry in tqdm(zip(outputs,problems)) :
        prompt = output.prompt
        for i in range(num_responses):  
            answers.append({
                'prompt': entry['prompt'],
                'answer': output.outputs[i].text,
            })
    return answers

def main() -> None:
    args = parse_arguments()
    problems = []
    with open(args.input_path, encoding='utf-8') as f:
        problems = json.load(f)

    answer = generate_answer_by_vllim(problems, args.model_name_or_path, args.num_responses)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = args.output_name + '_num_' + str(args.num_responses) + '_time_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.json'
    
    with open(os.path.join(args.output_dir, output_name), mode='w', encoding='utf-8') as f:
        json.dump(answer, f, indent=5, ensure_ascii=False)

if __name__=='__main__':
    sys.exit(main())