# Generalization of Transformers with In-Context Learning: An Empirical Study

> Generalization of Transformers with In-Context Learning: An Empirical Study [ICLR 2025]
>
> Xingxuan Zhang, Haoran Wang, Jiansheng Li, Yuan Xue, Shikai Guan, Renzhe Xu, Hao Zou, Han Yu, Peng Cui

📝 [Paper](https://openreview.net/forum?id=yOhNLIqTEF)

The code repository contains code for all of our main experiments.

- embed-level function fitting
- token-level function fitting
- translation task
- api-calling task

As each task can be split into `preprocess`, `train` and `evaluation` clearly, you can read the comments of the corresponding source code to get usage.

For submodules, you can use `git submodule update --init` to get them.

Thanks for those repos:

- https://github.com/hiyouga/LLaMA-Factory/
- https://github.com/dtsip/in-context-learning/
- https://github.com/Blealtan/efficient-kan/
- https://github.com/THUNLP-MT/StableToolBench/
