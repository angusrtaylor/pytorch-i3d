import torch.optim as optim
from models.pytorch_i3d import InceptionI3d
import math


if __name__ == "__main__":
    model = InceptionI3d(400, in_channels=3)
    optimizer = optim.SGD(
        model.parameters(),
        lr=1.0,
        momentum=0.9, 
        weight_decay=0.0000001
    )
    #step_size = 25
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, mode='triangular2', step_size_up=25)
    #clr = cyclical_lr(step_size, min_lr=0.0001, max_lr=0.1, mode='triangular')
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    for i in range(200):
        print("step ", i, " ", optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()