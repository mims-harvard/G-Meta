import torch.nn as nn


class SGC(nn.Module):
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)


def get_model(model_opt, nfeat, nclass, cuda=True):
    if model_opt == 'SGC':
        model = SGC(nfeat=nfeat, nclass=nclass)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))
    if cuda:
        model.cuda()
    return model
