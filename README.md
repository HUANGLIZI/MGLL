# MGLL: Multi-Granular Language Learning

This repository is the official implementation of the paper *“Boosting Medical Visual Understanding From Multi-Granular Language Learning”*. [Arxiv](https://arxiv.org/abs/2511.15943)

 ## Abstract

Recent advances in image-text pretraining have significantly enhanced visual understanding by aligning visual and textual representations. Contrastive Language-Image Pretraining (CLIP) has played a pivotal role in multimodal learning. However, its focus on single-label, single-granularity alignment limits its effectiveness in complex domains such as medical imaging, where images often correspond to multiple labels across different levels of granularity. To address this, we propose Multi-Granular Language Learning (MGLL), a contrastive learning framework designed to improve both multi-label and cross-granularity alignment. MGLL leverages structured multi-label supervision, integrates textual descriptions across granularities, and introduces soft-label supervision with point-wise constraints to enhance alignment. MGLL employs smooth Kullback–Leibler (KL) divergence to ensure cross-granularity consistency while maintaining computational efficiency as a plug-and-play module for vision-language models. Pretrained on our constructed large-scale multi-granular datasets and evaluated across multiple datasets, MGLL outperforms other state-of-the-art methods in downstream tasks. The code will be available on GitHub.

## Requirements

Python == 3.11 and install from the `requirements.txt` using:

```
pip install -r requirements.txt
```

## Usage

### 1. Pre-training

You can set your parameters in `./exps/pretrain.sh` and train your own model by running the following command.

```bash
bash ./exps/pretrain.sh
```

### 2. Downstream

You can set your parameters in `./exps/downstream.sh` and train your own model by running the following command.

```bash
bash ./exps/downstream.sh
```

To obtain pre-trained models and the dataset, you can contact the email address zhanli@uw.edu. We just handle the **real-name email** and **your email suffix must match your affiliation**. The email should contain the following information:
```angular2html
Name/Homepage/Google Scholar: (Tell us who you are.)
Primary Affiliation: (The name of your institution or university, etc.)
Job Title: (E.g., Professor, Associate Professor, Ph.D., etc.)
Affiliation Email: (the password will be sent to this email, we just reply to the email which is the end of "edu".)
How to use: (Only for academic research, not for commercial use or second-development.)
```

