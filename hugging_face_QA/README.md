## re-init:
迁移原始的TensorFlow 1.15项目至基于torch的transformers

采用example里面的原始代码，迁移至两个框架里面；run_qa和run_seq2seq

一个是基于seq2seq的mt5模型，t5-base-Chinese

它是把encoder部分处理成["question:", _question.lstrip(), "context:", _context.lstrip()]的形式

TODO： 后面如果要做multi-task的话，可以考虑把这种比较单一的prompt做一些修改

一个是基于mrc的bert模型，chinese_pretrain_mrc_roberta_wwm_ext_large

由于从原来的TensorFlow版本迁移过来了，可以很方便地对一些已经训练好的mrc模型进行fine-tune

之前只是利用了bert语言模型的先验信息，因此Q的设计比较简单，直接把类似当成了一个无实际意义的prompt

考虑直接对chinese_pretrain_mrc_roberta_wwm_ext_large进行mrc提问

    question = "资金账户风险"
    context = "股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"

得到如下结果：

{'score': 0.04694908857345581, 'start': 98, 'end': 100, 'answer': '被盗'}

改用意图更加明显的prompt：

    question = "发生资金账户风险的是？"
    context = "股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"

得到如下结果：

{'score': 0.05301574245095253, 'start': 77, 'end': 80, 'answer': '酒鬼酒'}

而原来的直接拼接方式需要进行多轮训练以后才能达到：

{'score': 0.9985246658325195, 'start': 77, 'end': 80, 'answer': '酒鬼酒'}

因此：带有明显意图的prompt，显然更加合适，可以进一步缩小pretrain和fine-tune的差距

同时这也可以看作是小样本学习的一个范式，将整个过程视作是先在一个bert上，拿各种各样的mrc数据集作为middle task/pretrain task，学习出一个更好地理解文本的模型

然后再在这个模型上进行fine-tune，肯定能够获得比之前更好的泛化性以及random query效果

todo：修改query的表达

两个子任务的共同需求都是要修改prompt的逻辑