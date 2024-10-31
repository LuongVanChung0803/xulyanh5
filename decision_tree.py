import numpy as np
import pandas as pd

# Hàm tính Gini Index
def gini_index(groups, classes):
    total_instances = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / total_instances)
    return gini

# Chia dữ liệu theo thuộc tính
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Tìm thuộc tính và giá trị tốt nhất để chia
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Tạo cây quyết định CART
def build_cart_tree(train, max_depth, min_size, depth=0):
    left, right = train['groups']
    del(train['groups'])
    if not left or not right:
        return left or right
    if depth >= max_depth:
        return left or right
    if len(left) <= min_size:
        return left
    if len(right) <= min_size:
        return right
    node = get_split(train)
    node['left'] = build_cart_tree(node['left'], max_depth, min_size, depth+1)
    node['right'] = build_cart_tree(node['right'], max_depth, min_size, depth+1)
    return node

# Dự đoán cho cây quyết định
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Hàm tính thông tin Gain
def entropy(class_labels):
    total = len(class_labels)
    class_counts = {}
    for label in class_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    entropy_value = 0.0
    for count in class_counts.values():
        probability = count / total
        entropy_value -= probability * np.log2(probability)
    return entropy_value

def information_gain(parent, left_child, right_child):
    total = len(parent)
    p_left = len(left_child) / total
    p_right = len(right_child) / total
    return entropy(parent) - (p_left * entropy(left_child) + p_right * entropy(right_child))

# Xây dựng cây quyết định ID3
def build_id3_tree(dataset, attributes):
    class_labels = [row[-1] for row in dataset]
    if class_labels.count(class_labels[0]) == len(class_labels):
        return class_labels[0]
    if not attributes:
        return max(set(class_labels), key=class_labels.count)

    best_attr = max(attributes, key=lambda attr: max(
        information_gain(class_labels, 
                         [row for row in dataset if row[attr] == value], 
                         [row for row in dataset if row[attr] != value])
        for value in set(row[attr] for row in dataset)
    ))

    tree = {best_attr: {}}
    remaining_attrs = [attr for attr in attributes if attr != best_attr]
    for value in set(row[best_attr] for row in dataset):
        subtree = build_id3_tree(
            [row for row in dataset if row[best_attr] == value], 
            remaining_attrs
        )
        tree[best_attr][value] = subtree
    return tree
