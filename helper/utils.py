import torch
import time



def create_model_load_weights(model, evaluation=False, ckpt_path=None):
    if evaluation and ckpt_path:
        state_dict = torch.load(ckpt_path)
        if 'module' in next(iter(state_dict)):
            state_dict = Parallel2Single(state_dict)
        state = model.state_dict()
        state.update(state_dict)
        model.load_state_dict(state)
    return model


def create_model_load_weights_v2(model, evaluation=False, ckpt_path=None):
    if evaluation and ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        if 'module' in next(iter(state_dict)):
            state_dict = Parallel2Single(state_dict)
        state = model.state_dict()
        state.update(state_dict)
        model.load_state_dict(state)
    return model


def get_optimizer(model, learning_rate=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    return optimizer

def struct_time():
    # 格式化成2020-08-07 16:56:32
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return cur_time

def simple_time():
    cur_time = time.strftime("[%m-%d-%H]", time.localtime())
    return cur_time