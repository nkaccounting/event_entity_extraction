import json

import pandas as pd

df = pd.read_csv('train.csv', header=None, index_col=None, encoding='utf-8')
df = df.fillna("#")

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

with open(('VU_squad2.0_train.json'), 'w', encoding='utf-8') as fp:
    json.dump({
        'version': 'v2.0',
        'data': data
    }, fp, ensure_ascii=False, indent=2)
