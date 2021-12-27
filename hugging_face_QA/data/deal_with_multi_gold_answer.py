# 把答案处理成多个answer的形式，即一个问题有多个gold answer
import json

with open('./json/VU_squad2.0_all.json', 'r', encoding='utf8') as fp:
    json_data = json.load(fp)
    new_data = []
    occur_context = {}
    loc = 0
    for data in json_data['data']:
        if 'hasAns' in data['id']:
            # 没有出现
            if not occur_context.get(data['context']):
                occur_context[data['context']] = loc
                loc += 1
                new_data.append(data)
            else:
                index = occur_context.get(data['context'])
                text = data['answers']['text'][0]
                answer_start = data['answers']['answer_start'][0]
                new_data[index]['answers']['text'].append(text)
                new_data[index]['answers']['answer_start'].append(answer_start)

with open(('./json/VU_squad2.0_temp.json'), 'w', encoding='utf-8') as fp:
    json.dump({
        'version': 'v2.0',
        'data': new_data
    }, fp, ensure_ascii=False, indent=2)
