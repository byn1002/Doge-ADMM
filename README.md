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
- graph decomposition(B=1,2,3);
- Doge-ADMM algorithm
- compare time and effects of denosing;

## File
- The algorithm implementation for our project can be found in the **Algorithm** folder.
- The experiment-related code is located in the **Experiment** folder.
- The remaining files include reproductions of certain papers and code for generating plots.
