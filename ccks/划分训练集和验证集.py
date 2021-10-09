import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ccks.create_squad_json import creat_json

df = pd.read_csv('train.csv', header=None, index_col=None, encoding='utf-8')
df = df.fillna("#")

split_first = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
split_second = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9, random_state=1)

for train_index, test_index in split_first.split(df, df[2]):
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

# 小tips，第二次分割的时候，所有的操作要完全忘记前面的东西，完全基于新的变量去弄
# 索引的时候是train_data.iloc[train_index]，新的train_data是train_data_
for train_index, validate_index in split_second.split(train_data, train_data[2]):
    train_data_ = train_data.iloc[train_index]
    validate_data = train_data.iloc[validate_index]

creat_json(train_data_, 'train', True)
creat_json(validate_data, 'validate', True)
creat_json(validate_data, 'validate--all_answer', False)
creat_json(test_data, 'test', True)
creat_json(test_data, 'test--all_answer', False)

# 检验是否发生数据交叉
train_index_set = set(train_data_.index)
validate_index_set = set(validate_data.index)
test_index_set = set(test_data.index)

if train_index_set.intersection(validate_index_set) or train_index_set.intersection(
        test_index_set) or validate_index_set.intersection(test_index_set):
    print("发生数据交叉")
else:
    print("数据被隔离划分")
