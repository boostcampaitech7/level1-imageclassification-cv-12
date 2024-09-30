import torch
'''
    해당 함수는 데이터 배치를 거스텀 가능한 함수입니다. 해당 코드는 기본적인 utils에서 지원하는 배치를 활용하고 있습니다.


    Args : batch

    Retrun batch
'''

def collate_fn(batch):
    batch = torch.utils.data.dataloader.default_collate(batch)
    return batch
