# ODMixer: Fine-grained Spatial-temporal MLP for Metro Origin-Destination Prediction

Metro Origin-Destination (OD) prediction is a crucial yet challenging spatial-temporal prediction task in urban
computing, which aims to accurately forecast cross-station ridership for optimizing metro scheduling and enhancing overall
transport efficiency. Analyzing fine-grained and comprehensive
relations among stations effectively is imperative for metro OD
prediction. However, existing metro OD models either mix information from multiple OD pairs from the station’s perspective
or exclusively focus on a subset of OD pairs. These approaches
may overlook fine-grained relations among OD pairs, leading
to difficulties in predicting potential anomalous conditions. To
address these challenges, we analyze traffic variations from the
perspective of all OD pairs and propose a fine-grained spatialtemporal MLP architecture for metro OD prediction, namely
ODMixer. Specifically, our ODMixer has double-branch structure
and involves the Channel Mixer, the Multi-view Mixer, and the
Bidirectional Trend Learner. The Channel Mixer aims to capture
short-term temporal relations among OD pairs, the Multi-view
Mixer concentrates on capturing relations from both origin and
destination perspectives. To model long-term temporal relations,
we introduce the Bidirectional Trend Learner. Extensive experiments on two large-scale metro OD prediction datasets HZMOD
and SHMO demonstrate the advantages of our ODMixer.

# Train And Eval
```bash
python train.py
python eval.py
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