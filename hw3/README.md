# hw3

rsync -avz --delete -e "ssh -p 30470 -i /Users/guoxi/.ssh/id_rsa" pku0030@36.170.52.66:/home/pku0030/homework_3 /Users/guoxi/Coding/ML/pku-llm/hw3

## 基线运行



- 原始CPU代码
    - 4096 Bodies: average 0.076 Billion Interactions / second

- 基础的GPU加速
    - 4096 Bodies: average 32.457 Billion Interactions / second
- 内存优化的GPU加速
    - 4096 Bodies: average 195.084 Billion Interactions / second

## 性能分析与理解

- 哪几行代码的修改，使得`nbody_parallel.cu`比`01-nbody.cu`执行快？为什么？
- `nbody_shared.cu`比`nbody_parallel.cu`执行快多少倍，为什么？
- 哪几行代码的修改，使得`nbody_shared.cu`比`nbody_parallel.cu`执行快？为什么？
- `nbody_shared.cu`比`01-nbody.cu`执行快多少倍，为什么？

## 参数分析与优化

- 分析`nbody_parallel.cu`中`BLOCK_SIZE`的取值对性能的影响。
- 分析`nbody_shared.cu`中`BLOCK_SIZE`、`BLOCK_STRIDE`的取值对性能的影响。