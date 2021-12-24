<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Question answering

This folder contains several scripts that showcase how to fine-tune a 🤗 Transformers model on a question answering dataset,
like SQuAD. 

## Trainer-based scripts

The [`run_qa.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py),
[`run_qa_beam_search.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa_beam_search.py) and [`run_seq2seq_qa.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_seq2seq_qa.py) leverage the 🤗 [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) for fine-tuning.

### Fine-tuning BERT on SQuAD1.0

The [`run_qa.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py) script
allows to fine-tune any model from our [hub](https://huggingface.co/models) (as long as its architecture has a `ForQuestionAnswering` version in the library) on a question-answering dataset (such as SQuAD, or any other QA dataset available in the `datasets` library, or your own csv/jsonlines files) as long as they are structured the same way as SQuAD. You might need to tweak the data processing inside the script if your data is structured differently.

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks), if it doesn't you can still use the old version of the script which can be found [here](https://github.com/huggingface/transformers/tree/master/examples/legacy/question-answering).

Note that if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along the flag `--version_2_with_negative`.

This example code fine-tunes BERT on the SQuAD1.0 dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large)
on a single tesla V100 16GB.

```bash
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

Training with the previously defined hyper-parameters yields the following results:

```bash
f1 = 88.52
exact_match = 81.22
```

### Fine-tuning XLNet with beam search on SQuAD

The [`run_qa_beam_search.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa_beam_search.py) script is only meant to fine-tune XLNet, which is a special encoder-only Transformer model. The example code below fine-tunes XLNet on the SQuAD1.0 and SQuAD2.0 datasets.

#### Command for SQuAD1.0:

```bash
python run_qa_beam_search.py \
    --model_name_or_path xlnet-large-cased \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_device_eval_batch_size=4  \
    --per_device_train_batch_size=4   \
    --save_steps 5000
```

#### Command for SQuAD2.0:

```bash
export SQUAD_DIR=/path/to/SQUAD

python run_qa_beam_search.py \
    --model_name_or_path xlnet-large-cased \
    --dataset_name squad_v2 \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./wwm_cased_finetuned_squad/ \
    --per_device_eval_batch_size=2  \
    --per_device_train_batch_size=2   \
    --save_steps 5000
```

### Fine-tuning T5 on SQuAD2.0

The [`run_seq2seq_qa.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_seq2seq_qa.py) script is meant for encoder-decoder (also called seq2seq) Transformer models, such as T5 or BART. These
models are generative, rather than discriminative. This means that they learn to generate the correct answer, rather than predicting the start and end position of the tokens of the answer.

This example code fine-tunes T5 on the SQuAD2.0 dataset.

```bash
python run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --dataset_name squad_v2 \
  --context_column context \
  --question_column question \
  --answer_column answer \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/fine_tune_seq2seq_squad/
```

## Accelerate-based scripts

Based on the scripts `run_qa_no_trainer.py` and `run_qa_beam_search_no_trainer.py`.

Like `run_qa.py` and `run_qa_beam_search.py`, these scripts allow you to fine-tune any of the models supported on a
SQuAD or a similar dataset, the main difference is that this script exposes the bare training loop, to allow you to quickly experiment and add any customization you would like. It offers less options than the script with `Trainer` (for instance you can easily change the options for the optimizer or the dataloaders directly in the script), but still run in a distributed setup, on TPU and supports mixed precision by leveraging the [🤗 `Accelerate`](https://github.com/huggingface/accelerate) library. 

You can use the script normally after installing it:

```bash
pip install accelerate
```

then

```bash
python run_qa_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/debug_squad
```

You can then use your usual launchers to run in it in a distributed environment, but the easiest way is to run

```bash
accelerate config
```

and reply to the questions asked. Then

```bash
accelerate test
```

that will check everything is ready for training. Finally, you cna launch training with

```bash
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/debug_squad
```

This command is the same and will work for:

- a CPU-only setup
- a setup with one GPU
- a distributed training with several GPUs (single or multi node)
- a training on TPUs

Note that this library is in alpha release so your feedback is more than welcome if you encounter any problem using it.


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
