import random


def sgc_data_generator(features, labels, node_num, select_array, task_num, n_way, k_spt, k_qry):
    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []
    class1_idx = []
    class2_idx = []

    labels_local = labels.clone().detach()
    select_class = random.sample(select_array, n_way)

    for j in range(node_num):
        if (labels_local[j] == select_class[0]):
            class1_idx.append(j)
            labels_local[j] = 0
        elif (labels_local[j] == select_class[1]):
            class2_idx.append(j)
            labels_local[j] = 1

    for t in range(task_num):
        class1_train = random.sample(class1_idx, k_spt)
        class2_train = random.sample(class2_idx, k_spt)
        class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
        class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
        class1_test = random.sample(class1_test, k_qry)
        class2_test = random.sample(class2_test, k_qry)
        train_idx = class1_train + class2_train
        random.shuffle(train_idx)
        test_idx = class1_test + class2_test
        random.shuffle(test_idx)
        x_spt.append(features[train_idx])
        y_spt.append(labels_local[train_idx])
        x_qry.append(features[test_idx])
        y_qry.append(labels_local[test_idx])

    return x_spt, y_spt, x_qry, y_qry