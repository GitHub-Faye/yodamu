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
    # 1. 创建基于yaml配置的pose模型
    pose_model = YOLO('ultralytics/cfg/models/v12/yolov12n-pose.yaml')
    
    # 2. 加载预训练的检测模型
    det_model = YOLO('yolov12n.pt')
    
    # 3. 获取两个模型的状态字典
    det_state_dict = det_model.model.state_dict()
    pose_state_dict = pose_model.model.state_dict()
    
    # 4. 创建新的状态字典，只包含匹配的层
    new_state_dict = {}
    transferred_layers = 0
    total_layers = len(pose_state_dict)
    
    # 5. 遍历pose模型的所有参数
    for k in pose_state_dict.keys():
        # 检查参数是否存在于检测模型中且形状相同
        if k in det_state_dict and det_state_dict[k].shape == pose_state_dict[k].shape:
            new_state_dict[k] = det_state_dict[k]
            transferred_layers += 1
        else:
            # 使用pose模型的初始化参数
            new_state_dict[k] = pose_state_dict[k]
    
    # 6. 加载新的状态字典到pose模型
    pose_model.model.load_state_dict(new_state_dict)

    

    
    # 开始训练
    results = pose_model.train(
        model=model_yaml,
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
