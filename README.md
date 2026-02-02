# MGLL: Multi-Granular Language Learning

This repository is the official implementation of the paper *“Boosting Medical Visual Understanding From Multi-Granular Language Learning”* (ICLR 2026). [Arxiv](https://arxiv.org/abs/2511.15943), [ResearchGate](https://www.researchgate.net/publication/397824340_Boosting_Medical_Visual_Understanding_From_Multi-Granular_Language_Learning)

 ## Abstract

Recent advances in image-text pretraining have significantly enhanced visual understanding by aligning visual and textual representations. Contrastive Language-Image Pretraining (CLIP) has played a pivotal role in multimodal learning. However, its focus on single-label, single-granularity alignment limits its effectiveness in complex domains such as medical imaging, where images often correspond to multiple labels across different levels of granularity. To address this, we propose Multi-Granular Language Learning (MGLL), a contrastive learning framework designed to improve both multi-label and cross-granularity alignment. MGLL leverages structured multi-label supervision, integrates textual descriptions across granularities, and introduces soft-label supervision with point-wise constraints to enhance alignment. MGLL employs smooth Kullback–Leibler (KL) divergence to ensure cross-granularity consistency while maintaining computational efficiency as a plug-and-play module for vision-language models. Pretrained on our constructed large-scale multi-granular datasets and evaluated across multiple datasets, MGLL outperforms other state-of-the-art methods in downstream tasks.

## Requirements

Python == 3.11 and install from the `requirements.txt` using:

```
pip install -r requirements.txt
```

## Dataset and Pretrain Model Weights

MIDRC dataset: [link](https://data.midrc.org/)

MIMIC-CXR: [link](https://physionet.org/content/mimic-cxr/2.1.0/)

Chest-Xray14: [link](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)

MGLL-Fundus dataset: [Image](https://drive.google.com/drive/folders/1gN3lzHQFECIJH1XqnY9GA5pBWCM1N0Io), [Text](https://drive.google.com/drive/folders/18-7d9ohHKqLosvONKEwv5ug_sX7Jjkd1)

Pretrain model weights on MGLL-Fundus: [link](https://drive.google.com/drive/folders/1zaM7qVOgHdWVcf9mVe-WI8Vpaf0W2Md2)

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

To obtain more pre-trained models and the in-house datasets in the future, you can contact the email address zhanli@uw.edu. We just handle the **real-name email** and **your email suffix must match your affiliation**. The email should contain the following information:
```angular2html
Name/Homepage/Google Scholar: (Tell us who you are.)
Primary Affiliation: (The name of your institution or university, etc.)
Job Title: (E.g., Professor, Associate Professor, Ph.D., etc.)
Affiliation Email: (the password will be sent to this email, we just reply to the email which is the end of "edu".)
How to use: (Only for academic research, not for commercial use or second-development.)
```










