import torch

# 데이터셋의 배치를 결정하는 모듈이
def collate_fn(batch):
    batch = torch.utils.data.dataloader.default_collate(batch)
    return batch
