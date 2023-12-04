from functools import reduce

def count_params(model):
    params = 0
    for p in model.parameters():
        params += reduce(lambda x, y: x * y, p.shape)
    return params


