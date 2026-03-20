# LMSFNet

> ****
>
> [Paper Link]() | [Pretrained Models](链接: https://pan.baidu.com/s/14hYByGQJJDz0897-ZmMDfA?pwd=3uvy 提取码: 3uvy)

## Table of Contents
- [Abstract](#abstract)
- [Environment Setup](#environment-setup)
- [Training & Testing](#training--testing)
- [Pretrained Models](#pretrained-models)
- [Experimental Results](#experimental-results)
- [Citation](#citation)

## Abstract
![moxingtu](assets/Figure_2.jpg)

## Environment Setup
### 1.Prerequisites
- Python 3.8
- PyTorch 1.12.0
- CUDA 12.8 

### 2.Install Dependencies
1. First, activate your experimental virtual environment (refer to the previous environment export steps)
2. Install all dependencies with one click:
```bash
pip install -r requirements.txt
```

## 3.Training & Testing

```bash
python Train.py --train_root YOUR_TRAININGSETPATH  --val_root  YOUR_VALIDATIONSETPATH  --save_path YOUR_CHECKPOINTPATH
 & python Test.py --train_root YOUR_TRAININGSETPATH  --val_root  YOUR_VALIDATIONSETPATH  --save_path YOUR_CHECKPOINTPATH
```

## 4. Evaluation

- Change the file path to your GT and testing path in [eval.py](), then run it to get your evaluation results.

## 5. Results download

The prediction results of our SDRNet are stored on [BaiDu Drive]() please check.


## 6.Citation

