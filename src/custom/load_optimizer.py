from torch.optim import AdamW


def load_optimizer(model, lr, weight_decay, betas):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
