# Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2506.07903)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![X](https://img.shields.io/badge/X-000000?logo=x&logoColor=white&style=flat-square)](https://x.com/YuchenZhu_ZYC/status/1934700344707363180)


Welcome to the official repository for **Diffuse Everything** (accepted at ICML 2025), for  <span style="color:#A020F0">**mixed-type tabular data synthesis**</span>. For text-image joint generation, please see [Diffuse-Everything](https://github.com/KevinRojas1499/Diffuse-Everything).

![Demonstration](assets/promo_v2.png)

## Introduction
**Diffuse Everything** is a general framework for building multimodal diffusion models for data of mixed modality, with a minimum need for tokenizers/VAEs/extra encoders. 

**Diffuse Everything** is built on <span style="color:orange">denoising markov models</span>, a generalized notion of denoising diffusion that characterizes the process using notions of Markov processes and their generators. This enables us to derive training objectives with theoretical guarantees in a general and modular fashion.

![denoising](assets/denoising_markov_models.jpg)

Specifically, in **Diffuse Everything**, we allow the diffusion process on each state space to have a <span style="color:orange">decoupled, independent schedule </span>, allowing modalities to be noised at their own pace. This enables the following benefits:
- Enjoying any-to-any within a single model
- A new guidance mechanism for multimodal generation

![decoupled](assets/decoupled.jpg)


## Latest Update

[Jun 23, 2025] We have open sourced the code.

[May 01, 2025] Diffuse Everything was accepted to ICML 2025.


## Installation
1. Clone the repository
```
git clone [repository-url]
cd Diffuse-Everything-Tabular
```
2. Install the dependencies
```
conda create --name diff-every-tabular python=3.10
conda activate diff-every-tabular
pip install synthcity
pip install category_encoders
pip install opacus==1.5.2
pip install -r requirement.txt
```
Note: synthcity==0.2.12 or 0.2.11 should work, as long as opacus==1.5.2



## Dataset Preparation
To download the datasets from [UCI Machine Learning Repository](https://archive.ics.uci.edu/), run the following command with scripts taken from [Tabsync](https://github.com/amazon-science/tabsyn),
```
python download_dataset.py 
```
This should download the dataset to the directory ```data/```. We preprocess the data with
```
python process_dataset.py
```
The prepocessed dat will be saved to ```synthetic/```

## Training
To train the model, run
```
python train_and_sample.py --dataname <dataset_name> --num_epochs <num_epochs>
```
Acceptable options of <dataset_name> is ```adult```, ```beijing```, ```default```, ```magic```, ```news```, ```shoppers```. 

The default directory for results saving is ```./exp```

## Sampling
After the model is trained, we generate samples from the saved checkpoint using
```
python sample.py --dataname <dataset_name> --ckpt_path ${exp_dir}/model_${model_idx}.pt --save_path ${exp_dir}
```
where ```exp_dir``` is the results directory, ```model_idx``` is the epoch number of the saved checkpoint.

## Evaluation
To evaluate the models (for 20 times) and compute (most) metrics including ```shape```, ```trend```, ```MLE```, ```precision``` and ```recall```, run
```
python eval/eval_all_metrics.py --dataname <dataset_name> --ckpt_path <ckpt_path> 
```
where the command standalone contains 20 times of sampling and metric computation.


The missing special cases are to  evaluate ```MLE``` for dataset ```default``` and ```news```, please instead run the following script file instead:
```
bash run_eval_mle_default_news.sh
```
where ```dataset```, ```model_idx``` and ```exp_dir``` need to be specified at the top of the file.


## Citation
If you find our work and repo help, we would appreciate your citations :smiling_face_with_three_hearts:

```
@inproceedings{
    rojas2025diffuse,
    title={Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces},
    author={Kevin Rojas and Yuchen Zhu and Sichen Zhu and Felix X-F. Ye and Molei Tao},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=AjbiIcRt6q}
}
```
```
@article{rojas2025diffuse,
    title={Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces},
    author={Rojas, Kevin and Zhu, Yuchen and Zhu, Sichen and Ye, Felix X-F and Tao, Molei},
    journal={arXiv preprint arXiv:2506.07903},
    year={2025}
}
```

## Acknowledgement
Part of the repo and helper functions are taken from the [Tabsync](https://github.com/amazon-science/tabsyn) repo, and our used backbone are modified based on DiT. 

