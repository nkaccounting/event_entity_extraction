# 将原来的验证集中的gold answers补充进来
import json

new_data = []

with open('./json/VU_squad2.0_validate.json', 'r', encoding='utf8') as fp:
    json_data = json.load(fp)
    validation_set = set()
    for data in json_data['data']:
        if 'hasAns' in data['id']:
            validation_set.add(data['id'])
        else:
            new_data.append(data)

with open('./json/VU_squad2.0_temp.json', 'r', encoding='utf8') as fp:
    json_data = json.load(fp)

    for data in json_data['data']:
        if 'hasAns' in data['id']:
            if data['id'] in validation_set:
                new_data.append(data)

with open(('./json/VU_squad2.0_multi_val.json'), 'w', encoding='utf-8') as fp:
    json.dump({
        'version': 'v2.0',
        'data': new_data
    }, fp, ensure_ascii=False, indent=2)
