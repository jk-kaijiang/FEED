# FEED: Fairness-Enhanced Meta-Learning for Domain Generalization

This repository contains the coded needed to reporduce the results of [FEED: Fairness-Enhanced Meta-Learning for Domain Generalization](https://arxiv.org/abs/2411.01316).

In this README, we provide an overview describing how this code can be run.  If you find this repository useful in your research, please consider citing:

```latex
@article{jiang2024feed,
  title={FEED: Fairness-Enhanced Meta-Learning for Domain Generalization},
  author={Jiang, Kai and Zhao, Chen and Wang, Haoliang and Chen, Feng},
  journal={arXiv preprint arXiv:2411.01316},
  year={2024}
}
```

## Quick Start

### Train a transformation model:

In folder FEED/domainbed/munit/

```python
python train_munit.py --output_path /munit_result_path --input_path1 /munit_result_path/outputs/tiny_munit/checkpoints/ --input_path2 /munit_result_path/outputs/tiny_munit/checkpoints/ --env 0 --device 0 --dataset FairFace --step 12
```

### Train a classifier:

Copy the last checkpoint file in /munit_result_path/outputs/tiny_munit/checkpoints/ to FEED/domainbed/munit/saved_models/FairFace/0 as 0_cotrain_step1.pt

In folder FEED/

```python
python -m domainbed.scripts.train --output_dir /result_path --test_envs 0 --dataset FairFace
```