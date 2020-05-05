from typing import Tuple, List

import torch
import torch.nn.functional as F
import ipdb
from sklearn.metrics import f1_score


def copy_weights(from_model, to_model):
    """Copies the weights from one model to another model.

    # Arguments:
        from_model: Model from which to source weights
        to_model: Model which will receive weights
    """
    if not from_model.__class__ == to_model.__class__:
        raise (ValueError("Models don't have the same architecture!"))

    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()


def autograd_graph(tensor):
    """Recursively retrieves the autograd graph for a particular tensor.

    # Arguments
        tensor: The Tensor to retrieve the autograd graph for

    # Returns
        nodes: List of torch.autograd.Functions that are the nodes of the autograd graph
        edges: List of (Function, Function) tuples that are the edges between the nodes of the autograd graph
    """
    nodes, edges = list(), list()

    def _add_nodes(tensor):
        if tensor not in nodes:
            nodes.append(tensor)

            if hasattr(tensor, 'next_functions'):
                for f in tensor.next_functions:
                    if f[0] is not None:
                        edges.append((f[0], tensor))
                        _add_nodes(f[0])

            if hasattr(tensor, 'saved_tensors'):
                for t in tensor.saved_tensors:
                    edges.append((t, tensor))
                    _add_nodes(t)

    _add_nodes(tensor.grad_fn)

    return nodes, edges


def pairwise_loss(u_batch, i_batch_p, i_batch_n, hid_d):
    u_batch = u_batch.view(len(u_batch), 1, hid_d)
    i_batch_p = i_batch_p.view(len(i_batch_p), hid_d, 1)
    i_batch_n = i_batch_n.view(len(i_batch_n), hid_d, 1)

    out_p = torch.bmm(u_batch, i_batch_p)
    out_n = - torch.bmm(u_batch, i_batch_n)

    # sum_p = F.logsigmoid(out_p)
    # sum_n = F.logsigmoid(out_n)
    # loss_sum = - (sum_p + sum_n)

    loss_sum = - F.logsigmoid(out_p + out_n)
    loss_sum = loss_sum.sum() / len(loss_sum)

    return loss_sum


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def F1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    macro_f1 = torch.tensor(f1_score(labels.cpu().data.numpy(), preds.cpu().data.numpy(), average='macro')).cuda()
    micro_f1 = torch.tensor(f1_score(labels.cpu().data.numpy(), preds.cpu().data.numpy(), average='micro')).cuda()
    return micro_f1

def accuracy_nn(preds, labels):
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train_accuracy_multilabel(output, labels, idx):
    preds = output.max(1)[1]
    correct = 0
    total_num = 0
    for i in range(len(idx)):
        total_num += 1
        index = idx[i]
        if preds[index] == labels[i]:
            correct += 1

    return correct, total_num


def accuracy_multilabel(output, labels, idx):
    # print output.size()
    preds = output.max(1)[1]

    correct = 0
    total_num = 0
    for i in range(len(idx)):
        total_num += 1
        index = idx[i]
        if preds[index] in labels[i]:
            correct += 1

    return correct, total_num
