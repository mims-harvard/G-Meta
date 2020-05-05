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

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


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


    def forward(self, x_t, y_t, c_t, graphlets):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = len(x_t)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        


        #logits, _ = self.net(x_spt[0].to(device), c_spt[0].to(device), graphlets, vars=None)
        #loss = F.cross_entropy(logits, y_spt[0].to(device))
        #grad = torch.autograd.grad(loss, self.net.parameters())
        #fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        #fast_weights = self.net.parameters()

        for i in range(task_num):
            '''
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), graphlets, fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i].to(device))
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i].to(device)).sum().item()
                corrects[0] = corrects[0] + correct
            '''
            accs = []
            for k in range(1, self.update_step):
                logits, _ = self.net(x_t[i].to(device), c_t[i].to(device), graphlets)
                

                target_cpu = y_t[i].to('cpu')
                input_cpu = logits.to('cpu')
                n_support = self.k_spt

                def supp_idxs(c):
                    # FIXME when torch will support where as np
                    return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

                # FIXME when torch.unique will be available on cuda too
                classes = torch.unique(target_cpu)
                n_classes = len(classes)
                # FIXME when torch will support where as np
                # assuming n_query, n_target constants
                n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

                support_idxs = list(map(supp_idxs, classes))

                prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
                # FIXME when torch will support where as np
                query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

                query_samples = input_cpu[query_idxs]
                dists = euclidean_dist(query_samples, prototypes)

                log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

                target_inds = torch.arange(0, n_classes)
                target_inds = target_inds.view(n_classes, 1, 1)
                target_inds = target_inds.expand(n_classes, n_query, 1).long()

                loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
                _, y_hat = log_p_y.max(2)
                acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
                accs.append(acc_val.item())
                #loss = F.cross_entropy(logits, y_t[i].to(device))
                self.optim.zero_grad()
                loss_val.backward()
                self.optim.step()


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

        return accs


    def finetune(self, x_t, y_t, c_t, graphlets):
        """
        b: number of tasks
        setsz: the size for each task

        :param x_spt:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_spt:   [b, setsz]
        :param x_qry:   [b], where each unit is a mini-batch of subgraphs, i.e. x_spt[0] is a DGL batch of # setsz subgraphs
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = len(x_t)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        
        net = deepcopy(self.net)
        opt = optim.SGD(net.parameters(), self.update_lr)

        #logits, _ = self.net(x_spt[0].to(device), c_spt[0].to(device), graphlets, vars=None)
        #loss = F.cross_entropy(logits, y_spt[0].to(device))
        #grad = torch.autograd.grad(loss, self.net.parameters())
        #fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        #fast_weights = self.net.parameters()

        for i in range(task_num):
            '''
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q, _ = self.net(x_qry[i].to(device), c_qry[i].to(device), graphlets, fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i].to(device))
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i].to(device)).sum().item()
                corrects[0] = corrects[0] + correct
            '''
            accs = []
            for k in range(1, self.update_step):
                logits, _ = net(x_t[i].to(device), c_t[i].to(device), graphlets)
                

                target_cpu = y_t[i].to('cpu')
                input_cpu = logits.to('cpu')
                n_support = self.k_spt

                def supp_idxs(c):
                    # FIXME when torch will support where as np
                    return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

                # FIXME when torch.unique will be available on cuda too
                classes = torch.unique(target_cpu)
                n_classes = len(classes)
                # FIXME when torch will support where as np
                # assuming n_query, n_target constants
                n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

                support_idxs = list(map(supp_idxs, classes))

                prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
                # FIXME when torch will support where as np
                query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

                query_samples = input_cpu[query_idxs]
                dists = euclidean_dist(query_samples, prototypes)

                log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

                target_inds = torch.arange(0, n_classes)
                target_inds = target_inds.view(n_classes, 1, 1)
                target_inds = target_inds.expand(n_classes, n_query, 1).long()

                loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
                _, y_hat = log_p_y.max(2)
                acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
                accs.append(acc_val.item())

                #loss = F.cross_entropy(logits, y_t[i].to(device))
                opt.zero_grad()
                loss_val.backward()
                opt.step()

        return accs


def main():
    pass


if __name__ == '__main__':
    main()
