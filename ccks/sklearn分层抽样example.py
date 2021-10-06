import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])

# n_splits：分割迭代的次数，如果我们要划分训练集和测试集的话，将其设置为1即可；几种分法，设置成3就可以随机3种生成方法
# test_size：分割测试集的比例；
# random_state：设置随机种子；
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1)

#可以理解为x，y，y是输出标签；同时也是基于y去做的分层抽样
for train_index, test_index in split.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train, y_train)
    print(X_test, y_test)
