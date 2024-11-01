import json
from typing import Dict, List, Tuple
from bpe_utils import Tokenizer, get_stats, merge, render_token
from transformers import  GPT2Tokenizer

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # 初始词表为256，合并vocab_size - 256次后达到词表大小
        num_merges = vocab_size - 256

        # 输入文本预处理
        text_bytes = text.encode("utf-8") #编码为原始字节
        ids = list(text_bytes) # 整数列表，每个整数取值为0-255（对应一个字节）

        # 迭代地合并最常见的pair以创建新的token
        merges:Dict[Tuple[int, int], int] = {} # (int, int) -> int,合并字节对为新的token
        vocab:Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} # int -> bytes, 词表
        for i in range(num_merges):
            stats = get_stats(ids)
            # 找到最常见的pair
            pair = max(stats, key=lambda x: stats[x])
            # 新token的索引
            idx = 256 + i
            # 合并所有出现的pair为新的token
            ids = merge(ids, pair, idx)
            # 保存合并结果
            merges[pair] = idx
            # 更新词表，合并两个字节对象为一个新的字节对象
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids:List[int])->str:
        # 给定一个token列表，返回一个字符串
        # 首先拼接所有token，然后解码为utf-8字符串
        text_bytes = b" ".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text:str)->List[int]:
        # 给定一个字符串，返回一个token列表
        text_bytes = text.encode("utf-8") # 编码为原始字节
        ids = list(text_bytes) # 整数列表，每个整数取值为0-255（对应一个字节）
        while len(ids) >= 2:
            # 找到合并索引值最低的字符对
            # (由于构建词表时merge有先后顺序,encode时也要按照对应顺序合并，先合并索引值低的pair）
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # 微妙之处：如果没有更多的合并可用，键将导致每个字符对的值都是inf，min将返回一个inf
            if pair not in self.merges:
                break # 不再有其他可以合并的内容
            # 否则合并最佳字符对（合并索引最低）
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


    

if __name__ == '__main__':
    tokenizer = BasicTokenizer()
    # with open('bpe/manual.txt', 'r', encoding='utf-8') as f:
    #     text = f.read()
    # tokenizer.train(text, 1024, verbose=True)
    # tokenizer.save('bpe/checkpoints/manual_1024')

    tokenizer.load('bpe/checkpoints/manual_1024.model')

    # 加载GPT-2的tokenizer
    gpt2Tokenizer = GPT2Tokenizer.from_pretrained('./bpe/assets/gpt2_tokenizer')
    print("GPT-2 tokenizer已成功加载")
    test_text1 = '博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。'
    test_text2 = 'Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university.'
    
    encoded_text:list = [[[], []], [[], []]] # encoded_text:list
    decoded_text:list = [[None, None], [None, None]] # decoded_text:list
    # 使用GPT-2的tokenizer对文本进行编码解码
    encoded_text[0][0] = gpt2Tokenizer.encode(test_text1)
    decoded_text[0][0] = gpt2Tokenizer.decode(encoded_text[0][0])
    encoded_text[0][1] = [gpt2Tokenizer.decode(i) for i in gpt2Tokenizer.encode(test_text2)]
    decoded_text[0][1] = gpt2Tokenizer.decode(gpt2Tokenizer.encode(test_text2))

    # 使用我的tokenizer对文本进行编码解码
    encoded_text[1][0] = tokenizer.encode(test_text1)
    decoded_text[1][0] = tokenizer.decode(encoded_text[1][0])
    encoded_text[1][1] = tokenizer.encode(test_text2)
    decoded_text[1][1] = tokenizer.decode(encoded_text[1][1])

    print("GPT-2 tokenizer编码1后的文本：",len(encoded_text[0][0]), encoded_text[0][0])
    print("我的tokenizer编码1后的文本：",len(encoded_text[1][0]), encoded_text[1][0])
    print("==============================================")
    print("GPT-2 tokenizer解码1后的文本：", decoded_text[0][0])
    print("我的tokenizer解码1后的文本：", decoded_text[1][0])
    print("==============================================")
    print("GPT-2 tokenizer编码2后的文本：",len(encoded_text[0][1]), encoded_text[0][1])
    print("我的tokenizer编码2后的文本：",len(encoded_text[1][1]), encoded_text[1][1])
    print("==============================================")
    print("GPT-2 tokenizer解码2后的文本：", decoded_text[0][1])
    print("我的tokenizer解码2后的文本：", decoded_text[1][1])
    print("==============================================")
    print("正确性比较：", decoded_text[0][0] == decoded_text[1][0], decoded_text[0][1] == decoded_text[1][1])