import os
import random
import string
import time
import pprint

from jsonlines import jsonlines
from torchtext.data import Dataset, Field, RawField, Example, NestedField
from tqdm import tqdm
from transformers import PreTrainedTokenizer, MT5Tokenizer, MT5TokenizerFast
from torchtext.data import Iterator
from typing import List, Tuple, Dict, AnyStr, Optional

from moqa.db import PassageDB
from moqa.common import config
import logging

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=config.log_file,
    level=config.log_level)

MKQA = "data/mkqa/mkqa.json"  #
DB_PATH = "data/wiki/all_passage.db"  # (lang_id, title, passage)


def main():
    # this is for testing
    from moqa.generative.trainer import Trainer
    with PassageDB('data/wiki/demo.db') as db:
        tokenizer = Trainer.init_tokenizer('google/mt5-small', 'data/cache/transformers')
        data = MT5Dataset(datafile='data/mkqa/mkqa_dpr_da.jsonl',
                          tokenizer=tokenizer,
                          db_multi=db,
                          langs=['da'],
                          context_length=3,
                          use_cache=False,
                          cache_dir='data/cache/generative')
        data_iter = Iterator(data, shuffle=True, sort=False, batch_size=1, train=False, repeat=False, device='cpu')
        for batch in data_iter:
            print("Batch:")
            print(batch)


class MT5Dataset(Dataset):
    def __init__(self,
                 datafile: AnyStr,
                 tokenizer: PreTrainedTokenizer,
                 db_multi: PassageDB,
                 langs: List[str],
                 context_length,
                 max_len=None,
                 is_training=True,
                 include_golden_passage=True,
                 preprocessing_truncation="truncate_only_passages",
                 include_passage_masks=False,
                 use_cache=True,
                 init_examples=True,
                 cache_dir='data/cache/generative', **kwargs):

        self.langs = langs
        self.cache_dir = cache_dir
        self.datafile = datafile
        self.tokenizer = tokenizer
        self.db_multi = db_multi
        self.max_len = tokenizer.model_max_length if max_len is None else max_len
        self.is_training = is_training
        self.use_cache = use_cache
        self.context_size = context_length
        self.include_golden_passage = include_golden_passage
        self.include_passage_masks = include_passage_masks
        self.preprocessing_truncation = preprocessing_truncation

        fields = self.prepare_fields(tokenizer.pad_token_id)
        if not include_passage_masks and 'doc_mask' in fields:
            del fields['doc_mask']
        self.fields = fields
        self.fields_tuple = list(fields.items())

        if init_examples:
            if use_cache:
                preprocessed_f = self.create_preprocessed_name()
                if not os.path.exists(preprocessed_f):
                    logging.info(f"{preprocessed_f} not found! Creating new...")

                    s_time = time.time()
                    examples = self.get_example_list()

                    logging.info(f"Saving {len(examples)} examples")
                    self.save(preprocessed_f, examples)

                    logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")
                    s_time = time.time()
                    examples = self.load_iterable(fields, examples, include_passage_masks=include_passage_masks)
                else:
                    s_time = time.time()
                    examples = self.load(preprocessed_f, fields, include_passage_masks=include_passage_masks)
                logging.info(f"Dataset {preprocessed_f} loaded in {time.time() - s_time:.2f} s")
            else:
                s_time = time.time()
                raw_examples = self.get_example_list()
                examples = self.load_iterable(fields, raw_examples, include_passage_masks=include_passage_masks)
                logging.info(f"Dataset {self.datafile} loaded in {time.time() - s_time:.2f} s")

            super().__init__(examples, fields, **kwargs)

    def create_preprocessed_name(self):
        without_psg_suffix = f"_withoutpassages" if not self.include_golden_passage else ""
        with_psg_masks = "_with_passage_masks" if self.include_passage_masks else ""
        preprocessed_f_noext = os.path.join(self.cache_dir, os.path.basename(
            self.datafile)) + f"_mkqa" \
                              f"_C{self.context_size}" \
                              f"{with_psg_masks}" \
                              f"{without_psg_suffix}" \
                              f"_{self.preprocessing_truncation}"
        preprocessed_f = preprocessed_f_noext + ".jsonl"
        return preprocessed_f

    @staticmethod
    def save(preprocessed_f: string, raw_examples: List[Dict]):
        with jsonlines.open(preprocessed_f, "w") as wf:
            for e in tqdm(raw_examples, desc=f"Saving into {preprocessed_f}"):
                wf.write(e)

    @staticmethod
    def load(preprocessed_f: string, fields: List[Tuple[str, RawField]], **kwargs) -> List[Example]:
        with jsonlines.open(preprocessed_f, "r") as raw_examples:
            return MT5Dataset.load_iterable(fields, raw_examples, **kwargs)

    @staticmethod
    def load_iterable(fields, raw_examples, include_passage_masks=False):
        fields = list(fields.items())
        examples = []
        for e in tqdm(raw_examples, desc="Loading preprocessed data..."):
            example = MT5Dataset.torchtext_example(e, fields, include_passage_masks)
            examples.append(example)
        return examples

    @staticmethod
    def torchtext_example(e, fields, include_passage_masks, choose_random_target=False):
        target = e["target"] if not choose_random_target else random.choice(e["target"])
        _preprocessed_example = [
            e["id"],
            e["question"],
            e["answers"],
            e["sources"],
            [[1] * len(x) for x in e["sources"]],
            e.get("doc_masks", None),
            target,
            [1] * len(target)]
        if not include_passage_masks:
            del _preprocessed_example[-3]
        example = Example.fromlist(_preprocessed_example, fields)
        return example

    def get_example_list(self):
        with open(self.datafile, encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)

        examples = []
        with jsonlines.open(self.datafile) as fp:
            for idx, sample in tqdm(enumerate(fp), total=num_lines):  # TODO: parallelize?
                if self.is_training:
                    examples += self.process_sample(sample)
                else:
                    # Do not use same question with multiple answers in validation
                    examples += [self.process_sample(sample)[0]]

                if idx == 0:
                    logging.info("Example of input formats:")
                    src_example1 = " ".join(self.tokenizer.convert_ids_to_tokens(examples[0]["sources"][0]))
                    src_example2 = " ".join(self.tokenizer.convert_ids_to_tokens(examples[0]["sources"][1]))
                    logging.info("inputs 1:")
                    logging.info(src_example1)
                    logging.info("inputs 2:")
                    logging.info(src_example2)
                    if len(examples[0]["target"]) > 1:
                        possible_target = examples[0]["target"]
                        if type(possible_target) == list:
                            possible_target = possible_target[0]
                        target_example = " ".join(self.tokenizer.convert_ids_to_tokens(possible_target))
                        logging.info("target:")
                        logging.info(target_example)

        return examples

    @staticmethod
    def prepare_fields(pad_t):
        WORD_field = Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t)
        WORD_nested_field = NestedField(Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t))
        PAD_field = Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        PAD_nested_field = NestedField(Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0))
        MASK_nested_field = NestedField(Field(use_vocab=False, batch_first=True, sequential=True, pad_token=1.))
        fields = {
            'id'         : RawField(),
            'question'   : RawField(),
            'answers'    : RawField(),
            'lang'       : RawField(),
            'src'        : WORD_nested_field,
            'src_mask'   : PAD_nested_field,
            'doc_mask'   : MASK_nested_field,
            'target'     : WORD_field,
            'target_mask': PAD_field,
            }
        return fields

    @staticmethod
    def assemble_target_sequences(answers: List, tokenizer: PreTrainedTokenizer):
        target_sequences = []
        for ans in answers:
            # T5 does this in their official T5 closed-book open-QA code
            # see https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob
            # /master/notebooks/t5-trivia.ipynb#scrollTo=OjEonhK3zNRu&line=18&uniqifier=1
            # Remove incorrect spacing around punctuation for NQ (but we keep the same code for all datasets)
            ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
            ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
            ans = ans.replace("( ", "(").replace(" )", ")")
            ans = ans.replace("`` ", "\"").replace(" ''", "\"")
            ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
            target_sequence = tokenizer.encode(ans, add_special_tokens=True)
            if type(tokenizer) in [MT5Tokenizer, MT5TokenizerFast]:
                # T5 starts generation with pad token
                target_sequence = [tokenizer.pad_token_id] + target_sequence
            else:
                assert False, "Unsupported tokenizer"
            # check there is only one pad and only one eos token
            assert target_sequence.count(tokenizer.eos_token_id) == 1
            assert target_sequence.count(tokenizer.pad_token_id) == 1
            target_sequences.append(target_sequence)

        return target_sequences

    def assemble_input_sequences(self, question: List[int], passages: List[List[int]],
                                 titles: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], List[List[int]]]:
        inputs = []
        document_masks = []

        if type(self.tokenizer) in [MT5Tokenizer, MT5TokenizerFast]:
            question_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.question_special_token)
            passage_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.passage_special_token)
            title_special_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.title_special_token)
            for title, passage in zip(titles, passages):
                question_and_title = [question_special_token] + question + \
                                     [title_special_token] + title + [passage_special_token]
                # Izacard et al. paper says:
                # we retrieve 100 passages (unless said otherwise), and truncate them to 250 word pieces.
                if self.preprocessing_truncation == "truncate_only_passages":
                    # but for datasets with shorter question/answers, we truncate only passages (e.g. NQ)
                    document = passage[:self.max_len - 1] + [self.tokenizer.eos_token_id]
                    seq = question_and_title + document
                    document_mask = [0] * len(question_and_title) + \
                                    [1] * len(document)
                elif self.preprocessing_truncation == "truncate_whole_input":
                    seq = question_and_title + passage
                    seq = seq[:self.max_len - 1] + [self.tokenizer.eos_token_id]
                    document_mask = [0] * len(question_and_title) + [1] * (len(passage) + 1)
                    document_mask = document_mask[:self.max_len]
                else:
                    raise ValueError(f"Unknown preprocessing truncation option: {self.preprocessing_truncation}")
                assert len(seq) == len(
                    document_mask), f"Sequence length: {len(seq)}, passage mask length {len(document_mask)}"
                inputs.append(seq)
                document_masks.append(document_mask)
        else:
            assert False, "Unsupported tokenizer"

        return inputs, document_masks

    def process_sample(self, sample: dict):
        """
        Creates numericalized input from raw sample
        :param sample: raw sample dictionary
        :return: numericalized sample(s), note that there can be more, as there can be more answers (or one
        multi-span answer in case of NQ, treated as more answers)
        """
        assert type(self.tokenizer) in [MT5Tokenizer, MT5TokenizerFast], f"Unsupported Tokenizer {type(self.tokenizer)}"

        # get gt_index - index of golden passage, if available
        gt_index = None
        if "gt_index" in sample and sample["gt_index"] != -1:
            gt_index = sample["gt_index"]

        # if golden passage is not available, start with empty set of passages
        if gt_index is None or not self.include_golden_passage:
            # unknown ground truth
            selected_ids = []
            titles = []
            titles_raw = []

            top_k_passages_tokens = []
            top_k_passages_raw = []
        else:  # otherwise, initialize with golden passage
            selected_ids = [(gt_index, 'en')]

            title, passage = self.db_multi.get_doc_text(gt_index, 'en', columns=["title", "passage"])

            titles = [self.tokenizer.encode(title, add_special_tokens=False)]
            titles_raw = [title]

            golden_passage = " " + passage
            top_k_passages_tokens = [self.tokenizer.encode(golden_passage, add_special_tokens=False)]
            top_k_passages_raw = [golden_passage]

        # take rest of the passages as top-k, if available
        for neg_ind in sorted(sample['retrieval'], key=lambda x: x['score'], reverse=True):
            idx: int = neg_ind['id']
            lang: str = neg_ind['lang']
            if len(top_k_passages_tokens) == self.context_size:
                break
            # if passage is already included (e.g. gt during training)
            elif (idx, lang) in selected_ids:
                continue
            else:
                selected_ids.append((idx, lang))
                title, passage = self.db_multi.get_doc_text(idx, lang, columns=["title", "passage"])

                # sometimes, there can be duplicate passages inside text (e.g. DPR passages), remove these
                if title in titles_raw and passage in top_k_passages_raw:
                    continue

                titles.append(self.tokenizer.encode(title, add_special_tokens=False))
                titles_raw.append(title)

                passage = " " + passage
                tokenized_passage = self.tokenizer.encode(passage, add_special_tokens=False)
                top_k_passages_tokens.append(tokenized_passage)
                top_k_passages_raw.append(passage)

        assert len(top_k_passages_tokens) == self.context_size, \
            f"Passages: {len(top_k_passages_tokens)}, Context size: {self.context_size}"

        queries = sample['queries']
        answers = sample['answers']
        examples = []
        for lang in queries.keys():
            question = queries[lang] + " ?"
            question_tokens = self.tokenizer.encode(question, add_special_tokens=False)

            input_sequences, document_masks = self.assemble_input_sequences(question=question_tokens,
                                                                            passages=top_k_passages_tokens,
                                                                            titles=titles)
            answer = answers[lang]
            target_sequences = self.assemble_target_sequences(answers=answer, tokenizer=self.tokenizer)

            if not target_sequences:  # in test time
                example = {
                    "id"       : sample["example_id"],
                    "question" : queries[lang],
                    "lang"     : lang,
                    "answers"  : [],
                    "sources"  : input_sequences,
                    "doc_masks": document_masks,
                    "target"   : [self.tokenizer.pad_token_id],
                    }
                if not self.include_doc_masks:
                    del example["doc_masks"]
                examples.append(example)
            else:
                for answer, targetSequence in zip(answers[lang], target_sequences):
                    # useful for debugging
                    # rev_input = " ".join(tokenizer.convert_ids_to_tokens(inputSequence))
                    # rev_target = " ".join(tokenizer.convert_ids_to_tokens(targetSequence))
                    example = {
                        "id"       : sample["example_id"],
                        "question" : queries[lang],
                        "lang"     : lang,
                        "answers"  : answer,
                        "sources"  : input_sequences,
                        "doc_masks": document_masks,
                        "target"   : targetSequence,
                        }
                    if not self.include_doc_masks:
                        del example["doc_masks"]
                    examples.append(example)
        return examples


def debug():
    with PassageDB('../../data/wiki/demo.db') as db:
        from moqa.generative import Trainer
        tokenizer = Trainer.init_tokenizer('google/mt5-small', 'data/cache/transformers')
        data = MT5Dataset(datafile='../../data/mkqa/mkqa_dpr_da.jsonl',
                          tokenizer=tokenizer,
                          db_multi=db,
                          is_training=False,
                          include_golden_passage=False,
                          langs=['da'],
                          context_length=3,
                          use_cache=True,
                          cache_dir='../../data/cache/generative')
        data_iter = Iterator(data, shuffle=False, sort=False, batch_size=1, train=False, repeat=False, device='cpu')
        for i, batch in enumerate(data_iter):
            print("Batch:", i)
            print(batch)


if __name__ == "__main__":
    debug()
    # main()
