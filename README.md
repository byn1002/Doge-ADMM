# Doge-ADMM

This repo contains a Python implementation for the paper Doge-ADMM: A Highly Scalable Parallel Algorithm for Trend Filtering on Graphs by Yinan Bu, Dongrun Wu and Yihong Zuo.


## Installation
Simply run

```conda env create -f environment.yml```

which will create an environment with packages installed.


## References:
- [Trend Filtering on Graphs](https://arxiv.org/pdf/1410.7690)
- [A Highly Scalable Parallel Algorithm for Isotropic Total Variation Models](https://proceedings.mlr.press/v32/wangb14.pdf)
- [fasttf](https://www.stat.cmu.edu/~ryantibs/papers/fasttf.pdf)

## Overleaf:
- [Doge-ADMM](https://www.overleaf.com/1426874579gcqzzrxcxgdx#d47b7c)

## Log:
- graph decomposition(k=0,1,2);
- 给出显式解-> （用C跑实验）（用python的thread跑并行）
- compare time and effects of denosing;

## 饼（冰）
- 进行三维图像的去噪；
- 给fmri去噪：根据z轴距离赋权重；对于时间序列同样进行去噪；

## file
- doge_admm.py Algorithm
- experiment_1.ipynb Experiment
