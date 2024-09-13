import torch

# 데이터셋의 배치를 결정하는 모듈이다.
# 특별한 커스텀을 하지 않고 단순히 기본적인 배치 처리를 진행하지만 후에 배치 처리시 커스텀이 가능하도록 모둘화 되어있다.
def mnist_collate_fn(batch):
    batch = torch.utils.data.dataloader.default_collate(batch)
    return batch
