import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from data.creat_huggingface_json import creat_json

df = pd.read_csv('train.csv', header=None, index_col=None, encoding='utf-8')
df = df.fillna("#")

# 9:1划分数据
split_first = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)

for train_index, test_index in split_first.split(df, df[2]):
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

creat_json(train_data, 'train', True)
creat_json(test_data, 'validate', True)

# 检验是否发生数据交叉
train_index_set = set(train_data.index)
test_index_set = set(test_data.index)

if train_index_set.intersection(test_index_set):
    print("发生数据交叉")
else:
    print("数据被隔离划分")
