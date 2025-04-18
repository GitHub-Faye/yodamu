#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import yaml_load
import os

# 配置文件路径
data_yaml = 'ultralytics/cfg/datasets/face5.yaml'
model_yaml = 'ultralytics/cfg/models/v12/yolov12n-pose.yaml'

def main():
    # 加载预训练的YOLOv12n检测模型
    model = YOLO('yolov12n.pt')
    

    

    
    # 开始训练
    results = model.train(
        data=data_yaml,       # 数据集配置
        epochs=1,           # 训练轮数
        imgsz=640,            # 图像大小
        batch=2,             # 批次大小
        name='yolov12n_face', # 实验名称
        val=True,             # 启用验证
        patience=10,          # 早停耐心值
        save=True,            # 保存模型
        project='runs/train', # 项目名称
        lr0=0.01,             # 初始学习率
        task='pose',          # 任务类型
    )
    
    print("训练完成!")
    
if __name__ == "__main__":
    main()
