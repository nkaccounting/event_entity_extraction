from transformers import MT5ForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline

model = MT5ForConditionalGeneration.from_pretrained('fine_tune_seq2seq_squad')

tokenizers = AutoTokenizer.from_pretrained('fine_tune_seq2seq_squad')

pipeline = Text2TextGenerationPipeline(model=model, tokenizer=tokenizers)

input = "question: 资金账户风险 context: 股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"

res = pipeline(
    input
)

print(res)
