# encoding=utf-8
from transformers import BertForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

model = BertForQuestionAnswering.from_pretrained('../../chinese_pretrain_mrc_roberta_wwm_ext_large')

tokenizers = AutoTokenizer.from_pretrained('../../chinese_pretrain_mrc_roberta_wwm_ext_large')

pipeline = QuestionAnsweringPipeline(model=model, tokenizer=tokenizers)

question = "近1年指的是什么？"
context = '1.患者，男性，86岁；2.因“反复便血11月，再发伴乏力3月，腹胀4天。”入院，11月前无明显诱因下出现粘液血便，呈鲜红色，非喷射性出血，在本院治疗，行肠镜：“直肠占位”。病理提示“粘膜部分腺体高级别上皮内瘤变，局部考虑癌变。”根据患者家属意见行保守治疗。之后症状反复。曾多次在我院内科住院。末两次住院时间：末次住院时间：2017.12.30-2018.2.9（住院号：90073）。出院诊断：“1.直肠癌；2.椎-基底动脉供血不足；3.2型糖尿病；4.原发性高血压3级 很高危；5.胆囊切除术后；6.脑梗塞；7.帕金森病；8.肝硬化失代偿期 脾肿大伴脾功能亢进 9.老年性脑改变。”近4天因停服利尿剂。3.患者有“高血压病”7年，有“糖尿病”病史4年，有“肝硬化、脾肿大、脾功能亢进、血三系减少”近1年。4.查体：神清，精神软，T：36.5℃；P：83次/分；R：20次/分；BP：159/87mmHg。贫血貌，全身浅表淋巴结未及肿大，颈软，两肺呼吸音清，未闻及干湿性啰音，心率83次/分，律齐，未闻及病理性杂音，腹膨隆, 肝脾肋下未及，移动性浊音+，双下肢无浮肿，四肢肌力V级。5.辅助检查：2017-7-31头颅CT检查：1.双侧基底节区、侧脑室旁多发缺血、梗塞灶；2.老年性脑改变；DR：主动脉型心脏。心电图：窦性心律；房性传导延缓。10-12腹部彩超：1、肝硬化声像图 2、胆囊切除术后 3、脾肿大 4、双肾小结石，右肾囊肿可能，建议随访 。11-14头颅CT复查：1、双侧基底节区、侧脑室旁多发缺血、梗塞灶。2、老年性脑改变。与2017年7月31日比较CT无明显变化。'

res = pipeline(
    question=question,
    context=context,
    topk=10
)

print(res)
