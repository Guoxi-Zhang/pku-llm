import re, collections
from typing import Dict, List, Tuple
import unicodedata


def get_stats(ids:List[int], counts:Dict[Tuple[int, int], int]=None) -> Dict[Tuple[int, int], int]:
    """
    给出一个id列表，返回相邻 id对 的出现次数,允许传入一个counts字典，用于累加计数
    Params:
        ids (List[int]): list of ids
        counts (Dict[Tuple[int, int], int]): 相邻字符对及其出现频率的字典
    Return:
        counts (Dict[Tuple[int, int], int]): 相邻字符对及其出现频率的字典
    Example: 
        ids=[1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids:List[int], pair:Tuple[int, int], idx:int) -> List[int]:
    """
    合并所有连续出现的pair为新的token idx，返回新的ids
    Params:
        ids (List[int]): list of ids
        pair (Tuple[int, int]): 待合并的字符对
        idx (int): 新的token id
    Return:
        newids (List[int]): 合并后的ids
    Example: 
        ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # 如果ids[i]和pair[0]相等，且i不是最后一个元素，且ids[i+1]和pair[1]相等，那么就把这两个元素合并成新的idx
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def replace_control_characters(s: str) -> str:
    # 去除控制字符
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # 美化打印token，去除控制字符
    s = t.decode('utf-8', errors='replace')
    # s = replace_control_characters(s)
    return s


class Tokenizer:
    """Tokenizers基类"""

    def __init__(self):
        # 默认: 词表大小为256(所有字节)，无合并，无模式
        self.merges:Dict[Tuple[int, int], int] = {}
        self.pattern = ""
        self.special_tokens:Dict[str, int] = {} # e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text:str, vocab_size:int, verbose=False):
        # 从text训练一个大小为vocab_size的词表
        raise NotImplementedError

    def encode(self, text:str):
        # 编码一个字符串为一个token列表
        raise NotImplementedError

    def decode(self, ids:list):
        # 解码一个token列表为一个字符串
        raise NotImplementedError

    def _build_vocab(self):
        # 词表从merges中简单且确定性地派生
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix:str):
        """
        保存2个文件: file_prefix.vocab 和 file_prefix.model
        - model文件是关键的，用于load()
        - vocab文件只是一个供人类检查的漂亮打印版本
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # 写入版本、模式
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # 写入特殊token，首先是它们的数量，然后是每一个token
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # 写入合并字典
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # 写入词表
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # 注意：许多token可能是部分utf-8序列，无法解码为有效字符串。这里我们使用errors='replace'将其替换为替换字符�。
                # 这也意味着我们不可能在load()中使用.vocab，因为以这种方式解码是有损的操作！
                s = render_token(token)
                # 如果idx在inverted_merges中，那么这个token有子节点，渲染为一个合并的token
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # 否则，这是一个单字节token, 直接打印
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file:str):
        """导入模型文件"""
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # 读取合并字典
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


