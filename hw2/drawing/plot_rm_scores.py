import os
import json
import matplotlib.pyplot as plt

import os
import json
import matplotlib.pyplot as plt



def extract_scores(file_path):
    """
    从指定文件中提取 score 并返回排序后的列表
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        scores = [item['score'] for item in data]
    return sorted(scores)

def plot_scores(file_list):
    """
    绘制不同文件的 score 折线图，并为每条线设置 label（文件夹名）
    """
    plt.figure(figsize=(10, 6))  # 设置图的大小

    for file_path in file_list:
        # 提取文件夹名作为 label
        folder_name = file_path.split('/')[1]
        # 提取 score 并排序
        scores = extract_scores(file_path)
        # 保存折线图
        plt.plot(scores, label=folder_name)
    
    # 设置图的标题和标签
    plt.xlabel('Index')
    plt.ylabel('Score')
    plt.title('Scores from different folders')
    plt.legend()  # 显示图例
    plt.grid(True)
    # 保存图片
    plt.savefig('Alpaca2-7B_plot_rm_scores.png')



if __name__ == "__main__":
    # 示例文件列表
    file_list = [
        'rm_score_output/Alpaca2-7B_better_data_1/eval_data_with_score.json',
        'rm_score_output/Alpaca2-7B_worse_data_1/eval_data_with_score.json',
    ]
    
    plot_scores(file_list)