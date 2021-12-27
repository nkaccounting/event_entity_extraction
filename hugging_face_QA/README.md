## re-init:
迁移原始的TensorFlow 1.15项目至基于torch的transformers

采用example里面的原始代码，迁移至两个框架里面；run_qa和run_seq2seq

1.一个是基于seq2seq的mt5模型，t5-base-Chinese

它是把encoder部分处理成["question:", _question.lstrip(), "context:", _context.lstrip()]的形式

TODO： 后面如果要做multi-task的话，可以考虑把这种比较单一的prompt做一些修改

    (例如：我知道问题XXX的答案在XXX里，你能告诉我是什么吗？)

2.一个是基于mrc的bert模型，chinese_pretrain_mrc_roberta_wwm_ext_large

由于从原来的TensorFlow版本迁移过来了，可以很方便地对一些已经训练好的mrc模型进行fine-tune

之前只是利用了bert语言模型的先验信息，因此Q的设计比较简单，直接把类似当成了一个无实际意义的prompt

考虑直接对chinese_pretrain_mrc_roberta_wwm_ext_large进行mrc提问

    question = "资金账户风险"
    context = "股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"
    answer = {'score': 0.04694908857345581, 'start': 98, 'end': 100, 'answer': '被盗'}

尝试另一组：

    question = "涉嫌提前泄露"
    context = "股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"
    answer = {'score': 0.43882259726524353, 'start': 24, 'end': 29, 'answer': '未公开信息'}



改用意图更加明显的prompt：

    question = "发生资金账户风险的是？"
    context = "股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"
    answer = {'score': 0.05301574245095253, 'start': 77, 'end': 80, 'answer': '酒鬼酒'}



而原来的直接拼接方式需要进行多轮训练以后才能达到：

    question = "资金账户风险"
    context = "股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"
    answer = {'score': 0.9985246658325195, 'start': 77, 'end': 80, 'answer': '酒鬼酒'}

因此：带有明显意图的prompt，显然更加合适，可以进一步缩小pretrain和fine-tune的差距

同时这也可以看作是小样本学习的一个范式，将整个过程视作是先在一个bert上，拿各种各样的mrc数据集作为middle task/pretrain task，学习出一个更好地理解文本的模型

然后再在这个模型上进行fine-tune，肯定能够获得比之前更好的泛化性以及random query效果

对基于理解文本，具备mrc先验能力的模型进行fine-tune

    question = "涉嫌提前泄露"（这个query类型是原来不具备的事件类型）
    context = "股价连续涨停后大股东拟减持 双一科技涉嫌提前泄露未公开信息、炒作股价配合股东减持遭深交所问询恒泰艾普(300157)两高管涉嫌违规减持 瞒天过海1年后曝光酒鬼酒(000799)子公司账户近1亿存款被盗 已报案"
    answer = {'score': 0.984306812286377, 'start': 14, 'end': 18, 'answer': '双一科技'}

分析不难得出，模型具备了更好的泛化性，可以分析出未训练的样本


把之前的例子又拿出来试了一下

    question = ["重组失败", '高管负面', '我爱你蜜雪冰城甜蜜蜜', '业绩下滑', '营收爆增', '北京大学', '高管正面']
    context = '今年6月，经检察机关批准，广州警方以涉嫌组织、领导传销活动罪对云联惠公司实际控制人黄某等主要犯罪嫌疑人执行逮捕'

结果：

    {'score': 0.9999895095825195, 'start': 0, 'end': 0, 'answer': ''}
    {'score': 0.9474249482154846, 'start': 31, 'end': 36, 'answer': '云联惠公司'}
    {'score': 0.9995309114456177, 'start': 0, 'end': 0, 'answer': ''}
    {'score': 0.9999899864196777, 'start': 0, 'end': 0, 'answer': ''}
    {'score': 0.9999699592590332, 'start': 0, 'end': 0, 'answer': ''}
    {'score': 0.9992692470550537, 'start': 0, 'end': 0, 'answer': ''}
    {'score': 0.9500406980514526, 'start': 31, 'end': 36, 'answer': '云联惠公司'}
整体的no answer已经没有之前那么不可解释了

其次，对于模型来说，它仍然是无法区别高管正面和高管负面，对于正面的样本，在未训练的情况下还是无法泛化

从这里得知https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline，
当中的handle_impossible_answer=True才能在pipeline里面处理不可回答的问题

添加后修改前面部分内容

并尝试之前所说的并行处理模式

    信批违规 {'score': 0.9999861717224121, 'start': 0, 'end': 0, 'answer': ''}
    实控人股东变更 {'score': 0.9999833106994629, 'start': 0, 'end': 0, 'answer': ''}
    交易违规 {'score': 0.9999847412109375, 'start': 0, 'end': 0, 'answer': ''}
    涉嫌非法集资 {'score': 0.9988253116607666, 'start': 0, 'end': 0, 'answer': ''}
    不能履职 {'score': 0.9998643398284912, 'start': 0, 'end': 0, 'answer': ''}
    重组失败 {'score': 0.9999895095825195, 'start': 0, 'end': 0, 'answer': ''}
    评级调整 {'score': 0.9999899864196777, 'start': 0, 'end': 0, 'answer': ''}
    业绩下滑 {'score': 0.9999899864196777, 'start': 0, 'end': 0, 'answer': ''}
    涉嫌违法 {'score': 0.8807060122489929, 'start': 0, 'end': 0, 'answer': ''}
    财务造假 {'score': 0.9999914169311523, 'start': 0, 'end': 0, 'answer': ''}
    涉嫌传销 {'score': 0.9724830389022827, 'start': 31, 'end': 36, 'answer': '云联惠公司'}
    涉嫌欺诈 {'score': 0.9997234344482422, 'start': 0, 'end': 0, 'answer': ''}
    资金账户风险 {'score': 0.9999895095825195, 'start': 0, 'end': 0, 'answer': ''}
    高管负面 {'score': 0.9474249482154846, 'start': 31, 'end': 36, 'answer': '云联惠公司'}
    资产负面 {'score': 0.999852180480957, 'start': 0, 'end': 0, 'answer': ''}
    投诉维权 {'score': 0.9992921948432922, 'start': 0, 'end': 0, 'answer': ''}
    产品违规 {'score': 0.9995466470718384, 'start': 0, 'end': 0, 'answer': ''}
    提现困难 {'score': 0.9999375343322754, 'start': 0, 'end': 0, 'answer': ''}
    失联跑路 {'score': 0.9992998242378235, 'start': 0, 'end': 0, 'answer': ''}
    歇业停业 {'score': 0.9996524453163147, 'start': 0, 'end': 0, 'answer': ''}
    公司股市异常 {'score': 0.9999790191650391, 'start': 0, 'end': 0, 'answer': ''}

通过这种方式，cpu机器上用6s就输出了答案

人造了一条多事件、单一事件单一主体的例子

今年6月，经检察机关批准，广州警方以涉嫌组织、领导传销活动罪对云联惠公司实际控制人黄某等主要犯罪嫌疑人执行逮捕,据悉，万事达有限责任公司严重违法，财务数据造假明显，高层领导人无法履行职责，此公司已被暂停营业'


    信批违规 {'score': 0.9999599456787109, 'start': 0, 'end': 0, 'answer': ''}
    实控人股东变更 {'score': 0.9999704360961914, 'start': 0, 'end': 0, 'answer': ''}
    交易违规 {'score': 0.9999322891235352, 'start': 0, 'end': 0, 'answer': ''}
    涉嫌非法集资 {'score': 0.9970194697380066, 'start': 0, 'end': 0, 'answer': ''}
    不能履职 {'score': 0.9893969297409058, 'start': 0, 'end': 0, 'answer': ''}
    重组失败 {'score': 0.9999475479125977, 'start': 0, 'end': 0, 'answer': ''}
    评级调整 {'score': 0.9999723434448242, 'start': 0, 'end': 0, 'answer': ''}
    业绩下滑 {'score': 0.9993011951446533, 'start': 0, 'end': 0, 'answer': ''}
    涉嫌违法 {'score': 0.41602638363838196, 'start': 59, 'end': 68, 'answer': '万事达有限责任公司'}
    财务造假 {'score': 0.9489694237709045, 'start': 59, 'end': 68, 'answer': '万事达有限责任公司'}
    涉嫌传销 {'score': 0.7133474349975586, 'start': 31, 'end': 36, 'answer': '云联惠公司'}
    涉嫌欺诈 {'score': 0.9244632124900818, 'start': 0, 'end': 0, 'answer': ''}
    资金账户风险 {'score': 0.9999752044677734, 'start': 0, 'end': 0, 'answer': ''}
    高管负面 {'score': 0.6239625811576843, 'start': 31, 'end': 36, 'answer': '云联惠公司'}
    资产负面 {'score': 0.9977694749832153, 'start': 0, 'end': 0, 'answer': ''}
    投诉维权 {'score': 0.9980773329734802, 'start': 0, 'end': 0, 'answer': ''}
    产品违规 {'score': 0.9977399706840515, 'start': 0, 'end': 0, 'answer': ''}
    提现困难 {'score': 0.9997305870056152, 'start': 0, 'end': 0, 'answer': ''}
    失联跑路 {'score': 0.9988224506378174, 'start': 0, 'end': 0, 'answer': ''}
    歇业停业 {'score': 0.6984049081802368, 'start': 59, 'end': 68, 'answer': '万事达有限责任公司'}
    公司股市异常 {'score': 0.9998917579650879, 'start': 0, 'end': 0, 'answer': ''}


整体效果也非常不错
##evaluate结果

***** eval metrics *****
  eval_HasAns_exact      = 92.8293
  eval_HasAns_f1         = 97.0235
  eval_HasAns_total      =    1283
  eval_NoAns_exact       = 98.4332
  eval_NoAns_f1          = 98.4332
  eval_NoAns_total       =    1468
  eval_best_exact        = 95.8197
  eval_best_exact_thresh =     0.0
  eval_best_f1           = 97.7758
  eval_best_f1_thresh    =     0.0
  eval_exact             = 95.8197
  eval_f1                = 97.7758
  eval_samples           =    2759
  eval_total             =    2751

10轮训练下来，相比于之前还有进一步的提升，但是evaluate并不能全面的反映好坏

就像是考试得了100百分，不代表真的就没有知识薄弱了

实际上基于mrc的fine-tune能够更好地处理unk的内容

修改query的表达

两个子任务的共同需求都是要修改prompt的逻辑
