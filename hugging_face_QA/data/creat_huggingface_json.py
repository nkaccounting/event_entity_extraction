import os.path
import random

import pandas as pd

import json

question_type = ['信批违规', '实控人股东变更', '交易违规', '涉嫌非法集资', '不能履职', '重组失败', '评级调整', '业绩下滑', '涉嫌违法', '财务造假', '涉嫌传销', '涉嫌欺诈',
                 '资金账户风险', '高管负面',
                 '资产负面', '投诉维权', '产品违规', '提现困难', '失联跑路', '歇业停业', '公司股市异常']


def creat_json(df: pd.DataFrame, type: str, switcher: bool):
    data = []
    for i in df.itertuples():
        id = str(i[1])
        context = i[2]
        question = i[3]
        answers = i[4]

        if answers == "#":
            pass
        else:
            try:
                answer_start = context.index(answers)
                right_answers = {
                    "text": [answers],
                    'answer_start': [answer_start]
                }
                result = {
                    "id": id + "-hasAns",
                    "title": id,
                    "context": context,
                    "question": '发生'+question+'的是？',
                    "answers": right_answers
                }
                data.append(result)
                if switcher == True:
                    question_candidate = random.choice(question_type)
                    # 不能让提出no answer的问题等于当前问题
                    while question_candidate == question:
                        question_candidate = random.choice(question_type)
                    result = {
                        "id": id + "-hasNoAns",
                        "title": id,
                        "context": context,
                        "question": '发生'+question_candidate+'的是？',
                        "answers": {
                            "text": [],
                            "answer_start": []
                        }
                    }
                    data.append(result)
            except:
                result = {
                    "id": id + "-hasNoAns",
                    "title": id,
                    "context": context,
                    "question": '发生'+question_candidate+'的是？',
                    "answers": {
                        "text": [],
                        "answer_start": []
                    }
                }
                data.append(result)

    if not os.path.exists('./json'):
        os.makedirs('./json')

    with open(('./json/VU_squad2.0_{type}.json').format(type=type), 'w', encoding='utf-8') as fp:
        json.dump({
            'version': 'v2.0',
            'data': data
        }, fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    df = pd.read_csv('train.csv', header=None, index_col=None, encoding='utf-8')
    df = df.fillna("#")
    creat_json(df, 'all', True)
