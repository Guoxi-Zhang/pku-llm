import os
import json
import random
import matplotlib.pyplot as plt

color_list = ['#6F6F6F', '#547BB4', '#629C35', '#C0321A', '#DD7C4F', '#6C61AF']
title = 'DPO'
def extract_data(file_path):
    """
    从指定文件中提取 prompt 和 score，并返回字典 {prompt: score}
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        return {item['prompt']: item['score'] for item in data}

def sort_first_file(first_file_path):
    """
    提取第一个文件的 prompt 和 score，并按 score 排序
    """
    data = extract_data(first_file_path)
    # 按 score 排序
    sorted_prompts = sorted(data, key=lambda x: data[x])
    sorted_scores = [data[prompt] for prompt in sorted_prompts]
    return sorted_prompts, sorted_scores

def align_prompts(sorted_prompts, file_path):
    """
    根据第一个文件的 prompt 顺序，重新排列当前文件的 score
    """
    data = extract_data(file_path)
    # 按第一个文件的 prompt 顺序排列
    aligned_scores = [data.get(prompt, None) for prompt in sorted_prompts]
    return aligned_scores

def chose_color(len:int):
    # 随机选择len个颜色
    return random.sample(color_list, len)


def plot_aligned_scores(file_list):
    """
    绘制按 prompt 对齐的 score 折线图
    """
    # 提取第一个文件的 prompt 和 score，并排序
    sorted_prompts, sorted_scores = sort_first_file(file_list[0])
    random_color_list = chose_color(len(file_list))
    # 初始化图
    plt.figure(figsize=(12, 8))

    # 绘制第一个文件的 score
    plt.plot(sorted_scores, marker='o', label=file_list[0].split('/')[-2], color=random_color_list.pop())

    # 对齐后续文件的 score 并绘制
    for file_path in file_list[1:]:
        folder_name = file_path.split('/')[-2]
        aligned_scores = align_prompts(sorted_prompts, file_path)
        meta_data={
            'dpo_data_mean': sum(sorted_scores)/len(sorted_scores),
            'no_dpo_data_mean': sum(aligned_scores)/len(aligned_scores),
            'no_dpo_better_than_dpo_count': 0,
            'no_dpo_worse_than_dpo_count': 0
        }
        for i in range(len(aligned_scores)):
            if aligned_scores[i] > sorted_scores[i]:
                meta_data['no_dpo_better_than_dpo_count'] += 1
            elif aligned_scores[i] < sorted_scores[i]:
                meta_data['no_dpo_worse_than_dpo_count'] += 1
        plt.scatter(range(len(aligned_scores)), aligned_scores, label=folder_name, alpha=0.8, color=random_color_list.pop())

    # 设置图的标题和标签
    plt.xlabel('Prompt Index')
    plt.ylabel('Score')
    plt.title('Aligned Scores by Prompt:'+title)
    plt.legend()
    plt.grid(True)
    plt.savefig(title+'_aligned_scores.png')

    # 输出元数据
    print(f"{title} meta data: \n {meta_data['dpo_data_mean'] = } \n {meta_data['no_dpo_data_mean'] = } \n {meta_data['no_dpo_better_than_dpo_count'] = } \n {meta_data['no_dpo_worse_than_dpo_count'] = }")

if __name__ == "__main__":
    # 示例文件列表
    file_list = [
        'rm_score_output/'+title+'_better_data_1/eval_data_with_score.json',
        'rm_score_output/'+title+'_worse_data_1/eval_data_with_score.json',
    ]
    file_list1 = [
        'rm_score_output/dpo_1/eval_data_with_score.json',
        'rm_score_output/no_dpo_1/eval_data_with_score.json',
    ]
    
    plot_aligned_scores(file_list1)