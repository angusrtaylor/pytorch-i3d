import torch.optim as optim
from models.pytorch_i3d import InceptionI3d
import math


# def cyclical_lr(step_sz, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None, scale_md='cycles', gamma=1.):
#     if scale_func == None:
#         if mode == 'triangular':
#             scale_fn = lambda x: 1.
#             scale_mode = 'cycles'
#         elif mode == 'triangular2':
#             scale_fn = lambda x: 1 / (2.**(x - 1))
#             scale_mode = 'cycles'
#         elif mode == 'exp_range':
#             scale_fn = lambda x: gamma**(x)
#             scale_mode = 'iterations'
#         else:
#             raise ValueError(f'The {mode} is not valid value!')
#     else:
#         scale_fn = scale_func
#         scale_mode = scale_md
#     lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)

#     def rel_val(iteration, stepsize, mode):
#         cycle = math.floor(1 + iteration / (2 * stepsize))
#         x = abs(iteration / stepsize - 2 * cycle + 1)
#         if mode == 'cycles':
#             return max(0, (1 - x)) * scale_fn(cycle)
#         elif mode == 'iterations':
#             return max(0, (1 - x)) * scale_fn(iteration)
#         else:
#             raise ValueError(f'The {scale_mode} is not valid value!')
#     return lr_lambda

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