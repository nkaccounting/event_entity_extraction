import pandas as pd
from transformers import BertForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

model = BertForQuestionAnswering.from_pretrained('../../chinese_pretrain_mrc_roberta_wwm_ext_large')

tokenizers = AutoTokenizer.from_pretrained('../../chinese_pretrain_mrc_roberta_wwm_ext_large')

pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizers)

dataframe = pd.read_csv('./data/train.csv', header=None, index_col=None, encoding='utf-8')
dataframe = dataframe.fillna("#")
given_answers = []
score = []
for i in dataframe.itertuples():
    context = i[2]
    question = i[3]
    if question != '其他':
        question = "发生" + i[3] + "的是？"
        res = pipeline(
            question=question,
            context=context,
            # handle_impossible_answer=True,#无监督条件下不支持不可回答模式，否则会造成模型不会的，也是不可回答的
        )
        given_answers.append(res['answer'])
        print(res['answer'])
        score.append(res['score'])
    else:
        given_answers.append('其他')
        score.append(0)

dataframe['given_answers'] = pd.Series(given_answers)
dataframe['score'] = pd.Series(score)

dataframe.to_csv("./data/Unsupervise_method.csv", index=0)
