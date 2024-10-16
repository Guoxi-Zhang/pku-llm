import re, collections
from typing import Dict, Tuple


def init_vocab(text:str) -> Dict[str, int]:
    '''
    初始化词频字典
    Params:
        text (str): 输入的文本
    Return:
        vocab (Dict[str, int]): 词频字典
    '''
    # 初始化词频字典
    vocab = collections.defaultdict(int)
    # 去头去尾再根据空格split
    for word in text.strip().split():
        # 给list中每个元素增加空格（表示未被合并的token），并在最后增加结束符号，同时统计单词出现次数
        vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab:Dict[str, int]) -> Dict[Tuple[str, str], int]:
    '''
    统计相邻字符对的出现频率
    Params:
        vocab (Dict[str, int]): 词频字典
    Return:
        pairs (Dict[Tuple[str, str], int]): 相邻字符对及其出现频率的字典
    '''
    pairs: Dict[Tuple[str, str], int] = collections.defaultdict(int)
    
    for word,freq in vocab.items():
        
        # 遍历每一个word里面的symbol，去凑所有的相邻两个内容
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i],symbols[i+1])] += freq
    return pairs

def merge_vocab(pair:Tuple[str, str], v_in:Dict[str, int]) -> Dict[str, int]:
    '''
    合并字符对
    Params:
        pair (Tuple[str, str]): 要合并的字符对
        v_in (Dict[str, int]): 合并前的vocab
    Return:
        v_out (Dict[str, int]): 合并后的vocab
    '''
    v_out = {}
    # 把pair拆开，然后用空格合并起来，然后用\把空格转义
    bigram = re.escape(' '.join(pair))
    # 自定义一个正则规则, (?<!\S)bigram(?!\S) 只有前面、后面不是非空白字符(\S)，才匹配bigram，这样不会匹配到位于单词中间的pair
    p: re.Pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for v in v_in:
        # 遍历当前的vocabulary，找到匹配正则的v时，用合并的pair去替换变成新的pair new，如果没有匹配上，那就保持原来的。
        # 比如pair当前是'h'和'e'，然后遍历vocabulary，找到符合前后不是非空白字符，只有'h\ e'的时候，就把他们并在一起变成'he'
        new = p.sub(''.join(pair),v)
        # 然后新的合并的数量就是当前vocabulary里面pair对应的数量
        v_out[new] = v_in[v]
    return v_out

def get_tokens(vocab:Dict[str, int]) -> Dict[str, int]:
    '''
    统计每个token的出现频率
    Params:
        vocab (Dict[str, int]): 词频字典
    Return:
        tokens (Dict[str, int]): 每个token的出现频率
    '''
    tokens: Dict[str, int] = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


