import json
from typing import Dict
from bpe_utils import init_vocab, get_stats, merge_vocab, get_tokens

class Tokenizer:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.tokens: Dict[str, int] = {}

    def train(self, text:str, vocab_size:int):
        """
        Train the tokenizer using BPE algorithm.
        Params:
            text (str): string-type data used to run BPE.
            vocab_size (int): the size of final vocabulary.

        Return:
            None
        """
        self.vocab = init_vocab(text)
        # 当词表大小小于vocab_size时，继续合并
        while len(self.tokens) < vocab_size:
            pairs = get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs.items(), key=lambda item: item[1]) # 使用 items() 和 lambda 函数
            # 如果pair的频率小于2，则停止合并
            if best[1] < 2:
                break
            best = best[0]
            self.vocab = merge_vocab(best, self.vocab)
            new_token = ''.join(best)
            self.tokens[new_token] = pairs[best]

    def encode(self, text:str)->list:
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.

        Return:
            ids (list): list of integer-type tokens.
        """
        # 将词表中的子词按照长度从大到小排序
        sorted_tokens = sorted(self.tokens.keys(), key=len, reverse=True)
        
        words = text.strip().split()
        encoded = []
        
        for word in words:
            i = 0
            while i < len(word):
                match = None
                # 遍历排序好的词表，寻找匹配的子词
                for token in sorted_tokens:
                    if word[i:i+len(token)] == token:
                        match = token
                        break
                if match:
                    encoded.append(match)
                    i += len(match)
                else:
                    # 如果没有匹配的子词，直接使用单个字符
                    encoded.append(word[i])
                    i += 1
        
        return encoded

    def decode(self, ids:list)->str:
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.

        Return:
            text (str): string-type data.
        """
        # 将token列表中的每个token连接起来，并去掉最后一个</w>
        text = ''.join(ids)
        text = text.replace(' </w>', '')
        return text
    
    # def load_vocab(self, vocab_path:str):
    #     with open(vocab_path, 'r', encoding='utf-8') as f:
    #         self.vocab = json.load(f)
    
    # def load_tokens(self, tokens_path:str):
    #     with open(tokens_path, 'r', encoding='utf-8') as f:
    #         self.tokens = json.load(f) 
    

if __name__ == '__main__':
    tokenizer = Tokenizer()
    with open('bpe/manual.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer.train(text, 1024)
    
    test_text = '博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。'
    print(tokenizer.encode(test_text))
    print(tokenizer.decode(tokenizer.encode(test_text)))
    # # 以json格式保存词表
    # with open('bpe/checkpoints/tokens.json', 'w', encoding='utf-8') as f:
    #     json.dump(tokenizer.tokens, f, ensure_ascii=False)
    # # 以json格式保存词表
    # with open('bpe/checkpoints/vocab.json', 'w', encoding='utf-8') as f:
    #     json.dump(tokenizer.vocab, f, ensure_ascii=False)

    




