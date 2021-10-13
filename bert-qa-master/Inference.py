import collections
import itertools
import os
import sys
from typing import List, Callable

import tensorflow as tf

import modeling
import tokenization
from run_squad import (
    SquadExample,
    InputFeatures,
    RawResult,
    FeatureWriter,
    model_fn_builder,
    input_fn_builder,
    get_final_text,
    _get_best_indexes,
    _compute_softmax,
    _check_is_max_context
)

curr_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_dir, '..'))


class QAInput:
    def __init__(self, question: str, context: str) -> None:
        self.question = question
        self.context = context


def gen_predictions(all_examples, all_features, all_results, n_best_size,
                    max_answer_length, do_lower_case):
    """Generate final predictions and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            feature_null_score = result.start_logits[0] + \
                                 result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = feature_index
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=min_null_feature_index,
                start_index=0,
                end_index=0,
                start_logit=null_start_logit,
                end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        # TODO: consider using orig_doc_start vs. start_index, since it do need additional transform
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_char", "end_char"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                        pred.end_index + 1)]
                # This might be paragraph based offset
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                        orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                # NOTE: seems it only consider the "text" instead of the offset
                final_text, (orig_start_position, orig_end_position) = get_final_text(
                    tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                (orig_start_position, orig_end_position) = (-1, -1)
                seen_predictions[final_text] = True

            if final_text:
                # This is used for debug, and is character based (the `pred.start_index` is token based)
                first_sep_token_index = feature.tokens.index('[SEP]')
                tokens_to_start = feature.tokens[first_sep_token_index +
                                                 1:pred.start_index]
                tokens_to_start_text = " ".join(tokens_to_start)

                # De-tokenize WordPieces that have been split off.
                tokens_to_start_text = tokens_to_start_text.replace(" ##", "")
                tokens_to_start_text = tokens_to_start_text.replace("##", "")

                # Try to deal with [UNK]
                # "~" is just a random char that take 1 index
                # feature.tokens: ['[CLS]', '超', '过', '40', '%', '的', '实', '体', '或', '主', '体', '是', '？', '[SEP]', '17', '年', '2', '月', '，', '中', '韩', '之', '间', '发', '生', '[UNK]', '萨', '德', '[UNK]', '事', '件', '，', '此', '后', '事', '件', '持', '续', '发', '酵', '，', '导', '致', '1', '##h', '##17', '境', '内', '赴', '韩', '游', '人', '次', '首', '次', '出', '现', '大', '幅', '下', '滑', '，', '旅', '游', '人', '次', '同', '比', '下', '滑', '超', '过', '40', '%', '，', '减', '少', '了', '156', '万', '人', '次', '。', '[SEP]']
                tokens_to_start_text = tokens_to_start_text.replace(
                    '[UNK]', '~')

                # Clean whitespace
                tokens_to_start_text = tokens_to_start_text.strip()
                tokens_to_start_text = "".join(tokens_to_start_text.split())

                start_char = len(tokens_to_start_text)
                end_char = start_char + len(final_text)

                # NOTE: Because the `get_final_text()` only consider "text" instead of "offset"
                #       That is if we care about "exact span", we should use `start_char` instead of `orig_start_position`
                # assert (start_char, end_char) == (
                #     orig_start_position, orig_end_position)

                # NOTE: if exist [UNK] then they will not match
                # assert orig_text[start_char:end_char] == orig_text[orig_start_position:orig_end_position], \
                #     f'Found {orig_text[start_char:end_char]} {orig_text[orig_start_position:orig_end_position]}'
                # BUG: 记录3组入室时(T0)、给负荷量后(T1)、手术开始时(T2)、射频消融开始后10min(T3)、射频消融结束后10min(T4)的收缩压(SBP)、舒张压(DBP)、心率(HR)、Ramsay镇静评分。
                if orig_text[start_char:end_char] != orig_text[orig_start_position:orig_end_position]:
                    # if reach any exception, we roll back so that it can align with `final_text`
                    (start_char, end_char) = (
                        orig_start_position, orig_end_position)
            else:
                start_char = -1
                end_char = -1

            # debug
            # print((start_char, end_char),
            #       (orig_start_position, orig_end_position),
            #       orig_text[start_char:end_char],
            #       orig_text[orig_start_position:orig_end_position])

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    # NOTE: the `pred.start_index` is not match the original text
                    start_char=start_char,
                    end_char=end_char))

        # if we didn't include the empty option in the n-best, include it
        if "" not in seen_predictions:
            nbest.append(
                _NbestPrediction(
                    text="", start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    start_char=-1,
                    end_char=-1))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_char=-1, end_char=-1))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_char"] = entry.start_char
            output["end_char"] = entry.end_char
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        # predict "" iff the null score - the score of best non-null > threshold
        score_diff = score_null - best_non_null_entry.start_logit - (
            best_non_null_entry.end_logit)
        scores_diff_json[example.qas_id] = score_diff
        # FLAGS.null_score_diff_threshold
        if score_diff > 0.0:
            all_predictions[example.qas_id] = ""
        else:
            all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json, scores_diff_json


class Inference:
    def __init__(self, model_dir: str, bert_config_file: str, vocab_file: str,
                 do_lower_case: bool = True, predict_batch_size: int = 8,
                 max_seq_length: int = 512, max_query_length: int = 64, doc_stride: int = 128):
        self.model_dir = model_dir

        self.max_seq_length = max_seq_length
        # The maximum number of tokens for the question. Questions longer than
        # this will be truncated to this length.
        self.max_query_length = max_query_length
        # When splitting up a long document into chunks, how much stride to
        # take between chunks.
        self.doc_stride = doc_stride
        self.do_lower_case = do_lower_case

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

        run_config = tf.contrib.tpu.RunConfig(
            cluster=None,
            master=None,
            model_dir=model_dir)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=None,
            learning_rate=None,
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=False,
            use_one_hot_embeddings=False)

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=predict_batch_size)

    @staticmethod
    def _read_squad_examples(input_data: List[dict]) -> List[SquadExample]:
        """Read a SQuAD json file into a list of SquadExample."""

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)

        return examples

    def convert_examples_to_features(self, examples: List[SquadExample],
                                     output_fn: Callable[[InputFeatures], None]) -> None:
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000

        for (example_index, example) in enumerate(examples):
            query_tokens = self.tokenizer.tokenize(example.question_text)

            # Truncate query if it is too long
            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(
                        tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                start_position = None
                end_position = None

                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible)

                # Run callback
                output_fn(feature)

                unique_id += 1

    @staticmethod
    def _transform_into_squad_dict(data: List[QAInput]) -> List[dict]:
        """
        Join input with same context
        """
        squad_dict = {
            'paragraphs': []
        }
        _id = 0
        for context, items in itertools.groupby(data, key=lambda x: x.context):
            qas = []
            for item in items:
                qas.append({
                    'question': item.question,
                    'id': _id
                })
                _id += 1

            squad_dict['paragraphs'].append({
                'qas': qas,
                'context': context
            })
        return [squad_dict]

    def predict(self, data: List[QAInput]):
        input_data = self._transform_into_squad_dict(data)
        pred_examples = self._read_squad_examples(input_data)

        # TODO: customize FeatureToExample and input_fn_builder
        #       so that we don't need to "write tf_record"
        # feature_processor = FeatureToExample()
        pred_writer = FeatureWriter(
            filename=os.path.join(self.model_dir, "pred.tf_record"),
            is_training=False)
        pred_features = []

        # pred_tf_examples = []

        def append_feature(feature: InputFeatures) -> None:
            pred_features.append(feature)
            pred_writer.process_feature(feature)
            # tf_example = feature_processor.process_feature(feature)
            # pred_tf_examples.append(tf_example)

        self.convert_examples_to_features(
            examples=pred_examples,
            output_fn=append_feature)
        pred_writer.close()

        # debug
        # for feature in pred_features:
        #     print('*** Example ***')
        #     print('unique_id:', feature.unique_id)
        #     print('example_index:', feature.example_index)
        #     print('tokens:', feature.tokens)

        predict_input_fn = input_fn_builder(
            input_file=pred_writer.filename,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)

        all_results = []
        predict_raw_results = self.estimator.predict(
            predict_input_fn, yield_single_examples=True)
        # debug
        # predict_raw_results = list(predict_raw_results)
        for result in predict_raw_results:
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        # Make `n_best_size`, `max_answer_length` to be global variable
        return gen_predictions(pred_examples, pred_features, all_results,
                               n_best_size=20, max_answer_length=30, do_lower_case=self.do_lower_case)

    def predict_spans(self, data: List[QAInput]) -> List[dict]:
        """
        Predict and get span text directly
        """
        results = self.predict(data)
        final_results = []
        for item, res in zip(data, results[1].values()):
            # This is possible to have empty text,
            # but in order to match the input length,
            # we will still return them
            final_results.append({
                'value': res[0]['text'],
                'score': res[0]['probability'],
                'start_char': res[0]['start_char'],
                'end_char': res[0]['end_char'],
            })

        # text only
        # return [res for res in results[0].values()]
        return final_results


def inference_from_half():
    import os

    model_name = 'finetuned_squad_10_9'
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(curr_dir, model_name)
    pretrain_dir = os.path.join(
        curr_dir, '../pretrain_model/chinese_L-12_H-768_A-12')
    inference = Inference(model_dir,
                          os.path.join(pretrain_dir, 'bert_config.json'),
                          os.path.join(pretrain_dir, 'vocab.txt'))
    QAList = [
        QAInput('实控人股东变更', 'STCN解读:长园集团(600525)股权争夺战现“三国杀”格局理想固网股东减持74万股'),
        QAInput('评级调整', 'easy-forex：西班牙评级遭下调加剧恐慌会计调整致连亏海鸟发展(600634)*ST前股东疯狂减持'),
        QAInput('不能履职', '辽宁方大集团(000055)清仓东北制药(000597)新华社:郭文贵海航“爆料”调查 遥控“内鬼” 获取内部信息方硕科技总经理潘旭东辞职'),
    ]
    for qa in QAList:
        data = [
            qa
        ]
        results = inference.predict(data)
        # print(results)
        print(inference.predict_spans(data))


if __name__ == '__main__':
    inference_from_half()
