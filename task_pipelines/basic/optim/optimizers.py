import torch
import inspect

try:
    import bitsandbytes as bnb
    adam8bit_class = bnb.optim.Adam8bit
except ImportError:
    adam8bit_class = None
    # pass, raise ImportErro

try:
    import prodigyopt
    prodigy_class = prodigyopt.Prodigy
except ImportError:
    prodigy_class = None

optimizer_dict = {'adam': torch.optim.Adam, 'adam8bit': adam8bit_class, 'adamw': torch.optim.AdamW, 'prodigy': prodigy_class}


def filter_valid_params(constructor, params_dict):
    valid_params = inspect.signature(constructor).parameters
    filtered_params = {key: value for key, value in params_dict.items() if key in valid_params}
    return filtered_params


if __name__ == '__main__':
    model1 = torch.nn.Linear(3,4)
    model2 = torch.nn.Conv2d(3, 6)
    # 메모리 줄이기
    model1_parameters = list(filter(lambda p: p.requires_grad, model1.parameters()))
    model1_parameters_with_lr = {"params": model1_parameters, "lr": 0.15}

    model2_parameters = list(filter(lambda p: p.requires_grad, model2.parameters()))
    model2_parameters_with_lr = {"params": model2_parameters, "lr": 0.1}


    params_to_optimizer = [model1_parameters_with_lr, model2_parameters_with_lr]
    my_params = {'betas' : (0.1, 0.1), 'weight_decay' : 0.99, 'eps':0.999, 'lr': 0.01}
    for key, optimizer_class in optimizer_dict.items():
        if optimizer_class is None:
            continue
        if key == 'adam8bit':   
            my_params['lr'] = 0.15
        valid_params = filter_valid_params(optimizer_class, my_params)
        print(valid_params)
        optimizer = optimizer_class(params_to_optimize, **valid_params)
        print(optimizer)