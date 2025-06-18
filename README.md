# ODMixer: Fine-grained Spatial-temporal MLP for Metro Origin-Destination Prediction (TKDE)
[Paper](https://arxiv.org/abs/2404.15734) 

## Abstract

Metro Origin-Destination (OD) prediction is a crucial yet challenging spatial-temporal prediction task in urban computing, which aims to accurately forecast cross-station ridership for optimizing metro scheduling and enhancing overall transport efficiency. Analyzing fine-grained and comprehensive relations among stations effectively is imperative for metro OD prediction. However, existing metro OD models either mix information from multiple OD pairs from the station's perspective or exclusively focus on a subset of OD pairs. These approaches may overlook fine-grained relations among OD pairs, leading to difficulties in predicting potential anomalous conditions. To address these challenges, we learn traffic evolution from the perspective of all OD pairs and propose a fine-grained spatial-temporal MLP architecture for metro OD prediction, namely ODMixer. Specifically, our ODMixer has double-branch structure and involves the Channel Mixer, the Multi-view Mixer, and the Bidirectional Trend Learner. The Channel Mixer aims to capture short-term temporal relations among OD pairs, the Multi-view Mixer concentrates on capturing spatial relations from both origin and destination perspectives. To model long-term temporal relations, we introduce the Bidirectional Trend Learner. Extensive experiments on two large-scale metro OD prediction datasets HZMOD and SHMO demonstrate the advantages of our ODMixer.

# Dataset
[百度网盘链接](https://pan.baidu.com/s/1OtLbMO4HE1W9hdbeE7zf8A)
提取码: cbia

# Train And Eval

```bash
# Train
python train.py --config_filename data/config/train_hz.yaml
python train.py --config_filename data/condfig/train_sh.yaml

# Eval
python eval.py --config_filename data/config/eval_hz.yaml
python eval.py --config_filename data/config/eval_sh.yaml
```

# Citation
```
@article{liu2024fine,
  title={Fine-grained Spatial-temporal MLP Architecture for Metro Origin-Destination Prediction},
  author={Liu, Yang and Chen, Binglin and Zheng, Yongsen and Li, Guanbin and Lin, Liang},
  journal={arXiv preprint arXiv:2404.15734},
  year={2024}
}
```