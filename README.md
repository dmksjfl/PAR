# Cross-Domain Policy Adaptation by Capturing Representation Mismatch (ICML 2024)

**Authors:** [Jiafei Lyu](https://dmksjfl.github.io/), [Chenjia Bai](https://baichenjia.github.io/), Jingwen Yang, [Zongqing Lu](https://z0ngqing.github.io/), Xiu Li

## Brief Introduction
We consider dynamics adaptation settings where there exists dynamics mismatch between the source domain and the target domain, and one can get access to sufficient source domain data, while can only have limited interactions with the target domain. We propose a *decoupled representation learning* approach for addressing this problem. We perform representation learning only in the target domain and measure the representation deviations on the transitions from the source domain, which we show can be a signal of dynamics mismatch. The produced representations are not involved in either policy or value function, but only serve as a reward penalizer. Our method achieves superior performance on environments with kinematic and morphology mismatch.

Please see our paper [here](https://arxiv.org/pdf/2405.15369) for more details.

## Method Overview

<img src="https://github.com/dmksjfl/PAR/blob/master/par.png" alt="image" width="600">

## Requirements

We run our experiments with Pytorch 1.8 and Gym version 0.23.1. You probably need other packages like `tensorboardX`, `numpy`. Please let me know if you have trouble running the experiments.

## How to run

To run this repo, you do not need to call `pip install -e .`. To reproduce our reported results in the submission, please check the following instructions:

For online PAR (online source domain and online target domain), run

```
CUDA_VISIBLE_DEVICES=0 python train.py --env halfcheetah_morph --beta 2.0 --seed 2 --dir logs
```

For offline PAR (offline source domain and online target domain), run

```
CUDA_VISIBLE_DEVICES=0 python train_offline.py --env halfcheetah --type medium --weight 5.0 --layernorm --seed 2 --dir logs
```

## Key Flags

For online PAR, specify the value of reward penalty coefficient $\beta$ by adding `--beta`. For offline PAR, specifying the dataset type adopted by `--type medium` and it supports medium, medium-replay, and medium-expert. One can also specify the value of $\nu$ by `--weight`.

## Citation

If you use our method or code in your research, please consider citing the paper as follows:
```
@inproceedings{lyu2024crossdomainpolicy,
 title={Cross-Domain Policy Adaptation by Capturing Representation Mismatch},
 author={Jiafei Lyu and Chenjia Bai and Jingwen Yang and Zongqing Lu and Xiu Li},
 booktitle={International Conference on Machine Learning},
 year={2024}
}
```
