# Generalization of Transformers with In-Context Learning: An Empirical Study

> Generalization of Transformers with In-Context Learning: An Empirical Study [ICLR 2025]
>
> Xingxuan Zhang, Haoran Wang, Jiansheng Li, Yuan Xue, Shikai Guan, Renzhe Xu, Hao Zou, Han Yu, Peng Cui

📝 [Paper](https://openreview.net/forum?id=yOhNLIqTEF) | 🤗 [Huggingface](https://huggingface.co/datasets/hrw/WMT500) | 📚 [Slide](https://docs.google.com/presentation/d/1BdGrJg4OUoUsbjLF0oXstOG67i5p5HzzLJ8NPoeIUjw/edit#slide=id.g33bf82ddc4f_2_48)

The code repository contains code for all of our main experiments.

- embed-level function fitting
- token-level function fitting
- translation
- api-calling

As each task can be split into `preprocess`, `train` and `evaluation` clearly, you can read the comments of the corresponding source code to get usage.

For submodules, you can use `git submodule update --init` to get them.

Thanks for those repos:

- https://github.com/hiyouga/LLaMA-Factory/
- https://github.com/dtsip/in-context-learning/
- https://github.com/Blealtan/efficient-kan/
- https://github.com/THUNLP-MT/StableToolBench/
