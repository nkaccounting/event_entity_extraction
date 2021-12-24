from transformers import BertForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

model = BertForQuestionAnswering.from_pretrained('../../chinese_pretrain_mrc_roberta_wwm_ext_large')

tokenizers = AutoTokenizer.from_pretrained('../../chinese_pretrain_mrc_roberta_wwm_ext_large')

pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizers)

question = "涉嫌提前泄露"
context = "股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"

res = pipeline(
    question=question,
    context=context
)

print(res)
