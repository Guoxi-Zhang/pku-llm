# path to your trained reward model
# for example: ~/output/rm
# MODEL_NAME_OR_PATH="~/models/Qwen-0.5B-Instruct" # model path
MODEL_NAME_OR_PATH="~/output/Alpaca-7B-train-1/slice_end" # model path

# for example: ~/align-anything/generate_scripts/test/Qwen-0.5B-Instruct_num_4_time_20241103_133249.json
# EVAL_DATASETS="/home/pku0030/align-anything/generate_scripts/test_no_dpo/Qwen-0.5B-Instruct_num_4_time_20241222_125252.json" # dataset path
# EVAL_DATASETS="/home/pku0030/align-anything/generate_scripts/test_dpo_1/slice_end_num_4_time_20241222_125821.json" # dataset path
EVAL_DATASETS="/home/pku0030/datasets/PKU-SafeRLHF-single-response/Alpaca3-8B/worse_data.json" # dataset path

EVAL_TEMPLATE="PKUSafeRLHF" # dataset template
EVAL_SPLIT="test" # split the dataset

OUTPUT_DIR="../rm_score_output/Alpaca3-8B_worse_data_1" # output dir

# For wandb online logging
export WANDB_API_KEY="6a5180fe8e7c65bbce02c94c2a9541970aceb877"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm_score \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \