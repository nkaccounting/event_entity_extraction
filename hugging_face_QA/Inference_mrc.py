import time

from transformers import BertForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

model = BertForQuestionAnswering.from_pretrained('./fine_tune_mrc_squad')

tokenizers = AutoTokenizer.from_pretrained('./fine_tune_mrc_squad')

pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizers)

# 测试之前的随机random query
question = ['信批违规', '实控人股东变更', '交易违规', '涉嫌非法集资', '不能履职', '重组失败', '评级调整', '业绩下滑', '涉嫌违法', '财务造假', '涉嫌传销', '涉嫌欺诈',
            '资金账户风险', '高管负面',
            '资产负面', '投诉维权', '产品违规', '提现困难', '失联跑路', '歇业停业', '公司股市异常']

question = ['发生' + q + '的是？' for q in question]

context = '今年6月，经检察机关批准，广州警方以涉嫌组织、领导传销活动罪对云联惠公司实际控制人黄某等主要犯罪嫌疑人执行逮捕,据悉，万事达有限责任公司严重违法，财务数据造假明显，高层领导人无法履行职责，此公司已被暂停营业'

t1 = time.time()
res = pipeline(
    question=question,
    context=context,
    handle_impossible_answer=True,
)
t2 = time.time()
for i, r in enumerate(res):
    print(question[i], r)

print(t2 - t1)
