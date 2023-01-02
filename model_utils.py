import torch

# src http://nlp.seas.harvard.edu/annotated-transformer/
def average(model, models):
    "Average models into model"
    for ps in zip(*[torch.tensor([p for p in m.parameters()]) for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
