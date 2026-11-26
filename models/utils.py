import torch

def stack_scores(scores):
    stacked_list = {}
    for out in scores:
        for key, val in out.items():
            if key in stacked_list:
                stacked_list[key].append(val)
            else:
                stacked_list[key] = [val]

    _template = {}
    for key, val in stacked_list.items():
        _template[key] = torch.stack(val).mean()

    return _template
