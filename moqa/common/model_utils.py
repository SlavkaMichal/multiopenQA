from torch import nn

def count_parameters(model: nn):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sum_parameters(model: nn):
    return sum(p.view(-1).sum() for p in model.parameters() if p.requires_grad)

def report_parameters(model: nn):
    num_pars = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    num_sizes = {name: p.shape for name, p in model.named_parameters() if p.requires_grad}
    return num_pars, num_sizes

