import torch
import torch.nn.functional as F

def cross_entropy(y_pred, y_true):
    # 定义一个小的正数epsilon，防止对数函数中出现log(0)导致无穷大的情况
    epsilon = 1e-9  
    # 使用clamp函数将预测概率限制在(epsilon, 1-epsilon)范围内，防止概率为0或1
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
    # 计算交叉熵损失，y_true * torch.log(y_pred)会将不相匹配的类别概率置为0
    loss = -torch.sum(y_true * torch.log(y_pred))
    # 返回计算得到的损失值
    return loss
