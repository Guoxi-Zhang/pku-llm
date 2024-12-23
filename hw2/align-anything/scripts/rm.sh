MODEL_NAME_OR_PATH="~/output/Alpaca-7B-train-1/slice_end" # model path
# MODEL_NAME_OR_PATH="../../models/Qwen-0.5B-Instruct" # model path

TRAIN_DATASETS="../../datasets/PKU-SafeRLHF-single-dimension/data/Alpaca-7B/train.json" # dataset path
TRAIN_TEMPLATE="PKUSafeRLHF" # dataset template
TRAIN_SPLIT="train" # split the dataset

EVAL_DATASETS="../../datasets/PKU-SafeRLHF-single-dimension/data/Alpaca3-8B/test.json" # dataset path
EVAL_TEMPLATE="PKUSafeRLHF" # dataset template
EVAL_SPLIT="test" # split the dataset

OUTPUT_DIR="~/output/Alpaca3-8B-eval-1" # output dir

# For wandb online logging
export WANDB_API_KEY="6a5180fe8e7c65bbce02c94c2a9541970aceb877"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_interval 1000000 \
     --epochs 3