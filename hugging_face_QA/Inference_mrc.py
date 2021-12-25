from transformers import BertForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

model = BertForQuestionAnswering.from_pretrained('./fine_tune_mrc_squad')

tokenizers = AutoTokenizer.from_pretrained('./fine_tune_mrc_squad')

pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizers)

# 测试之前的随机random query
question = ["重组失败", '高管负面', '我爱你蜜雪冰城甜蜜蜜', '业绩下滑', '营收爆增', '北京大学', '高管正面']
context = '今年6月，经检察机关批准，广州警方以涉嫌组织、领导传销活动罪对云联惠公司实际控制人黄某等主要犯罪嫌疑人执行逮捕'

res = pipeline(
    question=question,
    context=context
)

for r in res:
    print(r)
