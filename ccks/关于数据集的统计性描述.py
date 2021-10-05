import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('train.csv', header=None, index_col=None, encoding='utf-8')
df = df.fillna("#")


def staticstic_length(dataframe, in_index, out_index):
    # 答案的长度
    dataframe[out_index] = [len(answer) for answer in dataframe[in_index]]
    print(dataframe[out_index].describe())
    print(dataframe[out_index].value_counts())

    # 基于统计信息，最短1，最长27，按如下设置去画图
    # bins=np.arange(柱子个数+1)-0.5
    maxlen = dataframe[out_index].quantile(0.75) if dataframe[out_index].max() > 100 else dataframe[out_index].max()
    plt.hist(dataframe[out_index], bins=np.arange(maxlen + 1) - 0.5, edgecolor='black')
    plt.xticks([i for i in range(1, int(maxlen + 1), 1)])  # 根据分布频率手动设置x轴的刻度
    plt.show()


# 统计答案信息
staticstic_length(df, 3, 4)

# 统计问题信息，事件类型种类
staticstic_length(df, 2, 5)

# 统计文章长度
staticstic_length(df, 1, 6)


