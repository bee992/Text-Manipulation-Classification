# Text-Manipulation-Classification
8th solution for "Tianchi-ICDAR 2023 DTT in Images 1: Text Manipulation Classification" competition.

## 技术路线
1. 利用EfficientNet B6作为特征提取器，将提取到的特征用MLP做二分类
2. 同理，利用NextViT作为特征提取器
3. 融合两个模型的结果，最终复赛排名第8

## 提分点
1. 数据预处理很重要：由于输入图像size不一致，输入时需要resize，此时resize的大小会对模型的指标带来很大影响
2. 数据增强很重要，在其他CV分类任务中常用的一些增强手段在该任务中应慎用or使用较小的阈值来约束，保证训练数据的稳定性
3. 外部数据集预训练