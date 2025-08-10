import torch
from torch import nn

class CenterLoss(nn.Module):
    def __init__(self, class_num, feat_num) -> None:
        super().__init__()
        self.cls_num = class_num
        self.feat_num = feat_num
        self.center = nn.Parameter(torch.randn(self.cls_num, self.feat_num)) # 中心点随机产生

    def forward(self, x, labels):
        center_exp = self.center.index_select(dim=0, index=labels.long()) # [N, 2]
        count = torch.histc(labels.float(), bins=self.cls_num, min=0, max=self.cls_num-1) # [10]
        count_exp = count.index_select(dim=0, index=labels.long())+1 # [N]
        # loss = torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(x - center_exp,2), dim=1)), count_exp)) # 求损失, 原公式
        loss = torch.sum(torch.div(torch.sum(torch.pow(x - center_exp,2), dim=1), 2*count_exp)) # 求损失，略不同
        return loss