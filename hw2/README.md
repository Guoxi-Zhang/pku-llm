# hw2

同步代码：
`rsync -avz --delete -e "ssh -p 30470 -i /Users/guoxi/.ssh/id_rsa" pku0030@36.170.52.66:/home/pku0030/align-anything /Users/guoxi/Coding/ML/pku-llm/hw2`

全部的git提交记录如下：

![image-20250110213451537](README/image-20250110213451537.png)

## 部署

已按要求完成部署

## 奖励模型实现

### 训练奖励模型

#### 偏好数据集键值转换

代码如下所示，可以完成数据集键值转化

```Python
@register_template('HOMEWORK')
class HOMEWORK(Template):
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
    user_prompt: str = 'USER: {input} '
    assistant_prompt: str = 'ASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = ''

    def format_supervised_sample(self， raw_sample: dict[str， Any]) -> dict[str， Any]:
        text = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(input=raw_sample['prompt'])}"
            f"{self.assistant_prompt.format(output=raw_sample['answer'])}"
        )

        prompt = (
            f'{self.system_prompt}'
            f"{self.user_prompt.format(input=raw_sample['prompt'])}"
            f"{self.assistant_prompt.format(output='')}"
        )
        
        return {
            'text': text，
            'prompt': prompt，
        }

    def format_preference_sample(self， raw_sample: dict[str， Any]) -> dict[str， Any]:
        metrics = raw_sample['better_response_id']
        better_response = raw_sample[f'response_{int(metrics)}']
        worse_response = raw_sample[f'response_{1-int(metrics)}']
        prompt = raw_sample['prompt']

        formatted_better_output = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=better_response)}'
        )
        formatted_worse_output = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output=worse_response)}'
        )

        return {
            'better_text': formatted_better_output，
            'worse_text': formatted_worse_output，
        }

    def check_equal(self， raw_sample: dict[str， Any]) -> bool:
        return False

    def format_prompt_only_sample(self， raw_sample: dict[str， Any]) -> dict[str， Any]:
        prompt = raw_sample['prompt']

        formatted_prompt = (
            f'{self.system_prompt}'
            f'{self.user_prompt.format(input=prompt)}'
            f'{self.assistant_prompt.format(output="")}'
        )

        return {'text': formatted_prompt}

```

#### 训练奖励模型

训练集表现如下：

![image-20241222161432377](README/image-20241222161432377.png)

验证集表现如下：

![image-20241222165040941](README/image-20241222165040941.png)

#### 评测奖励模型
在对不同版本的Alpaca模型进行评测时，观察到了以下特征：

- 训练集表现：
  - 模型在训练集上的accuracy接近1，loss接近0，表明模型在训练数据上拟合得很好
  - 但这种过于完美的训练表现，结合测试集上只有70%左右的准确率，明显说明模型出现了过拟合现象
  - 过拟合导致模型在训练集上表现优异，但在新数据上泛化能力受限

- 测试集表现：
  - 准确率方面：三个模型的准确率都在70%左右，相比训练集的接近100%有明显下降
  - 奖励均值方面：三个模型的奖励均值差异较大，从-0.06到0.085不等
  - 奖励标准差方面：三个模型都在3.6左右，表明输出稳定性相近

具体评测结果如下:

Alpaca-7B：Evaluation: accuracy = 0.709987， reward_mean = -0.059841， reward_std = 3.661636

![image-20241222170720292](README/image-20241222170720292.png)

Alpaca2-7B：Evaluation: accuracy = 0.705611， reward_mean = 0.084601， reward_std = 3.672186

![image-20241222170546420](README/image-20241222170546420.png)

Alpaca3-8B：Evaluation: accuracy = 0.717497， reward_mean = 0.004012， reward_std = 3.615102

![image-20241222171001668](README/image-20241222171001668.png)

#### 使用奖励模型可视化偏好数据集
使用奖励模型给每个数据集的所有回答评分后，统计评分情况，包含四个关键指标：
- `better_data_mean`: 被标记为"更好"回答的平均分数
- `worse_data_mean`: 被标记为"更差"回答的平均分数
- `worse_better_than_better_count`: "更差"回答得分高于"更好"回答的次数
- `worse_worse_than_better_count`: "更差"回答得分低于"更好"回答的次数

在偏好数据集中，**训练后的模型能够更好地识别和生成符合数据集偏好的答案**，这在使用奖励模型给数据集的评分分析中可以看出：
- better answer的平均分数更高。
- "更差"回答得分低于"更好"回答的次数更多。

这表明模型在区分回答质量方面的能力有所提升。



具体的数据与各个数据集可视化结果（散点在曲线之上表明数据集中worse answer评分高于better answer，反之则相反）如下：

- Alpaca-7B

    - meta data:

        ```Python
        meta_data['better_data_mean'] = -0.14017175900179568 
        meta_data['worse_data_mean'] = -1.7415155766145238 
        meta_data['worse_better_than_better_count'] = 891 
        meta_data['worse_worse_than_better_count'] = 2017
        ```

    - ![image-20241223113917703](README/image-20241223113917703.png)

- Alpaca2-7B

    - meta data:

        ```Python
        meta_data['better_data_mean'] = 0.09921623596992159 
        meta_data['worse_data_mean'] = -1.5838211826958393 
        meta_data['worse_better_than_better_count'] = 826 
        meta_data['worse_worse_than_better_count'] = 1915
        ```

    - ![image-20241223113754745](README/image-20241223113754745.png)

- Alpaca3-8B

    - meta data:

        ```Python
        meta_data['better_data_mean'] = 0.005225250489152403 
        meta_data['worse_data_mean'] = -1.7122713441193689 
        meta_data['worse_better_than_better_count'] = 629 
        meta_data['worse_worse_than_better_count'] = 1608
        ```

    - ![image-20241223114107442](README/image-20241223114107442.png)


### 回答问题

- 奖励建模有哪些应用？
  - 识别并区分不同输出文本之间的优劣，为后续的强化学习阶段提供准确的奖励信号，指导模型生成更符合人类偏好的输出
  - 评估LLM生成文本的质量，作为自动评测的指标
  - 可以用于过滤和筛选LLM生成的候选回答
- 奖励建模训练可能存在哪些鲁棒性问题？
    - **对抗样本攻击：**精心构造的输入可能导致奖励模型给出错误评分，影响其性能
    - **分布外泛化：**在训练数据分布之外的样本上，模型表现可能不稳定
    - **标注噪声：**人类标注可能存在不一致的现象，影响模型学习
    - **长度偏差：**模型倾向于对较长/较短的回答给出偏差的评分，影响评价效果
- 如何缓解奖励建模的长度偏差？
  - 确保训练数据中包含不同长度的样本，避免模型偏向某一长度。
  - 在评分时对文本长度进行归一化处理，消除长度对评分的影响。
  - 在损失函数中加入长度惩罚项，防止模型过度依赖长度特征。
  - 对输入文本进行截断或填充，使其达到固定长度，减少长度差异。
- 有哪些拟合多元人类偏好的奖励建模方法？
  - 多任务学习: 同时学习多个偏好维度的评分
  - 分层奖励建模: 将整体偏好分解为多个子偏好，分别建模后再综合。
  - 条件奖励模型: 根据不同场景/用户动态调整偏好权重

## DPO微调

### 使用DPO微调模型

#### 运行DPO微调

![image-20241222202519655](README/image-20241222202519655.png)

#### 评测DPO微调模型

DPO微调结果如下：

- DPO 

    - meta data: 

        ``` python
        meta_data['dpo_data_mean'] = 5.8768333830040875 
        meta_data['no_dpo_data_mean'] = 7.0823259483566074 
        meta_data['no_dpo_better_than_dpo_count'] = 263 
        meta_data['no_dpo_worse_than_dpo_count'] = 103
        ```

    - ![image-20241223114826742](README/image-20241223114826742.png)

从数据可以看出，使用DPO微调后的模型效果反而不如原始模型:

1. 平均得分下降: DPO微调后的模型平均得分为5.88，而原始模型的平均得分为7.08，说明整体生成质量有所下降

2. 对比胜率偏低: 在366个测试样本中，DPO模型仅在103个样本上表现更好，而在263个样本上表现更差，胜率仅为28%

效果不佳的原因可能是:

1. 训练数据质量问题: DPO依赖于高质量的偏好数据，如果训练数据中的偏好标注存在噪声或不一致，会影响模型学习效果
2. 训练不充分: 可能需要更多的训练轮次或更大的训练数据集来让模型充分学习人类偏好
4. 评估指标偏差: 评估指标可能与DPO的优化目标不完全一致， 导致评估结果不能准确反映模型的实际改进

### 回答问题

- 从强化学习的角度，DPO是off-policy且offline的方法:
  - Off-policy: DPO不需要与环境交互来收集新的数据，而是直接使用已有的偏好数据进行训练
  - Offline: DPO完全基于静态的历史数据集进行训练，不需要在训练过程中收集新数据

- DPO主要针对传统RLHF的以下方面进行优化:
  - 简化了训练流程，去除了PPO算法中的价值网络和在线采样过程
  - 将RLHF重新表述为一个分类问题，通过最大化偏好数据的似然来优化策略
  - 发现了RLHF的最优策略可以用一个封闭形式的解来表示，从而避免了迭代式的PPO训练

- DPO相比传统RLHF的局限性:
  - 只能处理成对的偏好数据，无法直接利用标量奖励信号
  - 对数据质量要求较高，需要大量高质量的人类偏好标注
  - 可能会过度拟合训练数据中的偏好，泛化性能有限
  - 缺乏对探索的明确建模，可能陷入局部最优

- 现有研究主要从以下方面优化DPO:
  - **KTO (KL-constrained Optimization)**: 通过引入KL散度约束，KTO在优化过程中防止模型过度拟合，同时保持生成结果的多样性。这种方法在平衡模型性能与生成多样性方面表现出色，尤其是在需要生成多样化输出的任务中
  - **SimPO (Simplified Policy Optimization)**: SimPO通过简化训练流程，减少了计算复杂度，从而提高了训练效率。通过减少不必要的计算步骤或引入更高效的优化算法来实现，适用于大规模数据集和复杂模型的训练
  - **ORPO (Online Reinforcement Policy Optimization)**: ORPO结合了在线学习机制，使模型能够持续从新数据中学习并动态调整策略。这种方法特别适用于数据流不断变化的场景，如实时推荐系统或动态环境中的决策任务
