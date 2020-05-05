import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Classifier
from    copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Classifier(config)
        self.net = self.net.to(device)
    
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.optim = optim.SGD(self.net.parameters(), self.update_lr)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, c_spt, graphlets):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """
        #task_num = len(x_spt)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        


        #logits, _ = self.net(x_spt[0].to(device), c_spt[0].to(device), graphlets, vars=None)
        #loss = F.cross_entropy(logits, y_spt[0].to(device))
        #grad = torch.autograd.grad(loss, self.net.parameters())
        #fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        #fast_weights = self.net.parameters()

        #for i in range(task_num):
            

        for k in range(1, self.update_step):
            logits, _ = self.net(x_spt.to(device), c_spt.to(device), graphlets)
            loss = F.cross_entropy(logits, y_spt.to(device))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            with torch.no_grad():
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y_spt.to(device)).sum().item()  # convert to numpy
                corrects[k] = corrects[k] + correct


        # end of all tasks
        # sum over all losses on query set across all tasks
        #loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        #self.meta_optim.zero_grad()
        #loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        #   print(torch.norm(p).item())
        #self.meta_optim.step()

        #print(querysz *task_num)
        accs = np.array(corrects) / len(x_spt)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry, c_spt, c_qry, graphlets):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        querysz = len(y_qry[0])

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        opt = optim.SGD(net.parameters(), self.update_lr)

        x_spt = x_spt[0]
        y_spt = y_spt[0]
        x_qry = x_qry[0]
        y_qry = y_qry[0]
        c_spt = c_spt[0]
        c_qry = c_qry[0]
        
        for k in range(1, self.update_step_test):
            logits, _ = net(x_spt.to(device), c_spt.to(device), graphlets)
            loss = F.cross_entropy(logits, y_spt.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

            logits_q, _ = net(x_qry.to(device), c_qry.to(device), graphlets)
            loss_q = F.cross_entropy(logits_q, y_qry.to(device))

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry.to(device)).sum().item()  # convert to numpy
                corrects[k] = corrects[k] + correct

        del net

        accs = np.array(corrects) / querysz

        return accs


def main():
    pass


if __name__ == '__main__':
    main()
