# LMSFNet

> **LMSFNet: Enhancing Camouflaged Object Detection via Lightweight Multi-Scale Structural and Semantic Fusion**
>
> [Paper Link]() | [Zenodo Archive]() | [Pretrained Models](https://xxx.xxx/your-model-link)

## Table of Contents
- [Abstract](#abstract)
- [Network Architecture](#network-architecture)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training & Testing](#training--testing)
- [Pretrained Models](#pretrained-models)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)

## Abstract
Camouflaged object detection (COD) remains a challenging task due to the high visual similarity between targets and their background, weak boundary cues, and significant scale variations. Existing methods often suffer from structural degradation caused by repeated downsampling, semantic inconsistencies in multi-scale fusion, and insufficient sensitivity to small objects. To address these issues, we propose LMSFNet, a lightweight multi-scale fusion network designed to enhance detection accuracy while maintaining computational efficiency. Our approach integrates three key modules: a Multi-Scale Small-object Structure Enhancement Module (MSSEM) to restore fine-grained details and weak boundaries; a Cross-Scale Feature Interaction (CFI) module to align semantic information across scales and mitigate fusion conflicts; and a Global-Local Context Fusion (GLCF) module to jointly model long-range dependencies and local structures. Experiments on three benchmark datasets—CAMO,COD10K,and NC4K—demonstrate that LMSFNet achieves competitive performance with only 26.2M parameters and 21.6 GFLOPs, outperforming several state-of-the-art methods in terms of boundary continuity and small-object detection. Furthermore, LMSFNet exhibits strong generalisation capability in downstream tasks such as polyp segmentation and crack detection, underscoring its practical applicability. This work highlights the potential of lightweight, multi-scale fusion strategies for advancing COD in real-world scenarios.![moxingtu](assets/moxingtu.png)

## Environment Setup
### Prerequisites
- Python 3.8
- PyTorch 1.12.0
- CUDA 12.8 

### Install Dependencies
1. First, activate your experimental virtual environment (refer to the previous environment export steps)
2. Install all dependencies with one click:
```bash
pip install -r requirements.txt

```

