import json
import os.path

import pandas as pd


def creat_json(df: pd.DataFrame, type: str):
    data = []
    for i in df.itertuples():
        id = str(i[1])
        context = i[2]
        question = i[3]
        answers = i[4]

        is_impossible = False
        # 没有答案的情况
        if answers == "#":
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
        else:
            try:
                answer_start = context.index(answers)
                answers = [
                    {
                        "text": answers,
                        'answer_start': answer_start
                    }
                ]
                result = {
                    "title": id,
                    'paragraphs': [
                        {
                            'qas': [
                                {
                                    'question': question,
                                    'id': id,
                                    'answers': answers,
                                    'is_impossible': is_impossible
                                }
                            ],
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
