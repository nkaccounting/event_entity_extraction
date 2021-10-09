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

        is_impossible = False
        # 没有答案的情况，后来经过数据分析，发现这部分“累计错误”数据用到下游任务，实际意义不大
        # 直接训练会训练出其他->no answer的强表示，意义不是很大，最后选择舍弃
        if answers == "#":
            pass
        else:
            # 潜在的多个地方拼成一个答案的情况，bert-mrc-sl
            try:
                answer_start = context.index(answers)
                right_answers = [
                    {
                        "text": answers,
                        'answer_start': answer_start
                    }
                ]
                qas = []
                qas.append({
                    'question': question,
                    'id': id + '-hasAns',
                    'answers': right_answers,
                    'is_impossible': is_impossible
                })
                if switcher == True:
                    question_candidate = random.choice(question_type)
                    # 不能让提出no answer的问题等于当前问题
                    while question_candidate == question:
                        question_candidate = random.choice(question_type)
                    qas.append({
                        'question': question_candidate,
                        'id': id + "-hasNoAns",
                        'answers': [],
                        'plausible_answers': [],
                        'is_impossible': True
                    })
                result = {
                    "title": id,
                    'paragraphs': [
                        {
                            'qas': qas,
                            'context': context
                        }
                    ]
                }
            except:
                print('no answer！')
                is_impossible = True
                answers = []
                result = {
                    "title": id,
                    'paragraphs': [
                        {
                            'qas': [
                                {
                                    'question': question,
                                    'id': id,
                                    'answers': answers,
                                    'plausible_answers': [],
                                    'is_impossible': is_impossible
                                }
                            ],
                            'context': context
                        }
                    ]
                }
        data.append(result)

    if not os.path.exists('./json'):
        os.makedirs('./json')

    with open(('./json/VU_squad2.0_{type}.json').format(type=type), 'w', encoding='utf-8') as fp:
        json.dump({
            'version': 'v2.0',
            'data': data
        }, fp, ensure_ascii=False, indent=2)
