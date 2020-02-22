import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
import random

from args import get_citation_args
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy


def train_regression(model, train_features, train_labels, epochs, weight_decay, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    return model


def test_regression(model, test_features, test_labels):
    model.eval()
    output = model(test_features)
    return accuracy(output, test_labels)


def reset_array():
    class1_train = []
    class2_train = []
    class1_test = []
    class2_test = []
    train_idx = []
    test_idx = []


def main():
    args = get_citation_args()
    n_way = args.n_way
    train_shot = args.train_shot
    test_shot = args.test_shot
    step = args.step
    node_num = args.node_num
    iteration = args.iteration

    accuracy_meta_test = []
    total_accuracy_meta_test = []

    set_seed(args.seed, args.cuda)

    adj, features, labels = load_citation(args.dataset, args.normalization, args.cuda)

    if args.dataset == 'cora':
        class_label = [0, 1, 2, 3, 4, 5, 6]
        combination = list(combinations(class_label, n_way))
    elif args.dataset == 'citeseer':
        node_num = 3327
        iteration = 15
        class_label = [0, 1, 2, 3, 4, 5]
        combination = list(combinations(class_label, n_way))

    if args.model == 'SGC':
        features = sgc_precompute(features, adj, args.degree)

    for i in range(len(combination)):
        print('Cross_Validation: ',i+1)
        test_label = list(combination[i])
        train_label = [n for n in class_label if n not in test_label]
        print('Cross_Validation {} Train_Label_List {}: '.format(i+1, train_label))
        print('Cross_Validation {} Test_Label_List {}: '.format(i+1, test_label))
        model = get_model(args.model, features.size(1), n_way, args.cuda)

        for j in range(iteration):
            labels_local = labels.clone().detach()
            select_class = random.sample(train_label, n_way)
            print('Cross_Validation {} ITERATION {} Train_Label: {}'.format(i+1, j+1, select_class))
            class1_idx = []
            class2_idx = []
            for k in range(node_num):
                if(labels_local[k] == select_class[0]):
                    class1_idx.append(k)
                    labels_local[k] = 0
                elif(labels_local[k] == select_class[1]):
                    class2_idx.append(k)
                    labels_local[k] = 1
            for m in range(step):
                class1_train = random.sample(class1_idx,train_shot)
                class2_train = random.sample(class2_idx,train_shot)
                class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
                class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
                train_idx = class1_train + class2_train
                random.shuffle(train_idx)
                test_idx = class1_test + class2_test
                random.shuffle(test_idx)

                model = train_regression(model, features[train_idx], labels_local[train_idx], args.epochs, args.weight_decay, args.lr)
                acc_query = test_regression(model, features[test_idx], labels_local[test_idx])
                reset_array()

        torch.save(model.state_dict(), 'model.pkl')

        labels_local = labels.clone().detach()
        select_class = random.sample(test_label, 2)
        class1_idx = []
        class2_idx = []
        reset_array()
        print('Cross_Validation {} Test_Label {}: '.format(i + 1, select_class))

        for k in range(node_num):
            if (labels_local[k] == select_class[0]):
                class1_idx.append(k)
                labels_local[k] = 0
            elif (labels_local[k] == select_class[1]):
                class2_idx.append(k)
                labels_local[k] = 1

        for m in range(step):
            class1_train = random.sample(class1_idx, test_shot)
            class2_train = random.sample(class2_idx, test_shot)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            random.shuffle(test_idx)

            model_meta_trained = get_model(args.model, features.size(1), n_way, args.cuda).cuda()
            model_meta_trained.load_state_dict(torch.load('model.pkl'))

            model_meta_trained = train_regression(model_meta_trained, features[train_idx], labels_local[train_idx], args.epochs, args.weight_decay, args.lr)
            acc_test = test_regression(model_meta_trained, features[test_idx], labels_local[test_idx])
            accuracy_meta_test.append(acc_test)
            total_accuracy_meta_test.append(acc_test)
            reset_array()
        if args.dataset == 'cora':
            with open('cora.txt', 'a') as f:
                f.write('Cross_Validation: {} Meta-Test_Accuracy: {}'.format(i+1, torch.tensor(accuracy_meta_test).numpy().mean()))
                f.write('\n')
        elif args.dataset == 'citeseer':
            with open('citeseer.txt', 'a') as f:
                f.write('Cross_Validation: {} Meta-Test_Accuracy: {}'.format(i+1, torch.tensor(accuracy_meta_test).numpy().mean()))
                f.write('\n')
        accuracy_meta_test = []
    if args.dataset == 'cora':
        with open('cora.txt', 'a') as f:
            f.write('Dataset: {}, Train_Shot: {}, Test_Shot: {}'.format(args.dataset, train_shot, test_shot))
            f.write('\n')
            f.write('Total_Meta-Test_Accuracy: {}'.format(torch.tensor(total_accuracy_meta_test).numpy().mean()))
            f.write('\n')
            f.write('\n\n\n')
    elif args.dataset == 'citeseer':
        with open('citeseer.txt', 'a') as f:
            f.write('Dataset: {}, Train_Shot: {}, Test_Shot: {}'.format(args.dataset, train_shot, test_shot))
            f.write('\n')
            f.write('Total_Meta-Test_Accuracy: {}'.format(torch.tensor(total_accuracy_meta_test).numpy().mean()))
            f.write('\n')
            f.write('\n\n\n')

if __name__ == '__main__':
    main()
