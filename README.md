# Cross-Domain Policy Adaptation by Capturing Representation Mismatch

This is the codes for our novel method, Policy Adaptation by Representation Mismatch (PAR). Under review, please do not distribute.

## How to run

To run this repo, you do not need to call `pip install -e .`. We run our experiments with Pytorch 1.8 and Gym version 0.23.1.

To reproduce our reported results in the submission, please check the following instructions:

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