import os
import random
import string
import time

from jsonlines import jsonlines
from torchtext.data import Dataset, Field, RawField, Example, NestedField, Iterator
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers import MT5Tokenizer as Tokenizer
from transformers import MT5TokenizerFast as TokenizerFast
from typing import List, Tuple, Dict, AnyStr, Optional
from random import sample
from moqa.db import PassageDB
from moqa.common import config
import ipdb
# from moqa.datasets.preprocessor import MKQAPrep
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
                 preprocess,
                 model_name,
                 tokenizer: PreTrainedTokenizer,
                 db_multi: Optional[PassageDB],
                 langs: List[str],
                 context_length,
                 answer_limit=1,
                 max_len=None,
                 data_size=-1,  # if lower than zero than does nothing
                 is_training=True,
                 include_golden_passage=True,
                 only_gt_passages=False,
                 preprocessing_truncation="truncate_only_passages",
                 include_passage_masks=False,
                 use_cache=True,
                 cached_data_dir=None,
                 init_examples=True,
                 cache_dir='data/cache/generative', **kwargs):

        self.answer_limit = answer_limit
        self.langs = langs
        self.cache_dir = cache_dir
        self.datafile = datafile
        self.model_name = model_name
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.db_multi = db_multi
        self.max_len = tokenizer.model_max_length if max_len is None else max_len
        self.is_training = is_training
        self.use_cache = use_cache
        self.cached_data_dir = cached_data_dir
        self.context_size = context_length
        self.include_golden_passage = include_golden_passage
        self.only_gt_passages = only_gt_passages
        self.include_passage_masks = include_passage_masks
        self.preprocessing_truncation = preprocessing_truncation
        self.data_size = data_size

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
        if self.cached_data_dir is not None:
            return self.cached_data_dir
        without_psg_suffix = f"_withoutpassages" if not self.include_golden_passage else ""
        with_psg_masks = "_with_passage_masks" if self.include_passage_masks else ""
        model_name = f"_{self.model_name}" if self.model_name else ""
        gt_only = "_gt_only" if self.only_gt_passages else ""
        answer_limit = f"_answers_{self.answer_limit}" if self.answer_limit != -1 else ""
        preprocessed_f_noext = os.path.join(self.cache_dir, os.path.basename(
            self.datafile)) + f"_mkqa" \
                              f"_C{self.context_size}" \
                              f"{answer_limit}" \
                              f"{with_psg_masks}" \
                              f"{gt_only}" \
                              f"{without_psg_suffix}" \
                              f"_{self.preprocessing_truncation}" \
                              f"{model_name}"
        preprocessed_f = preprocessed_f_noext + ".jsonl"
        return preprocessed_f

    @staticmethod
    def save(preprocessed_f: string, raw_examples: List[Dict]):
        with jsonlines.open(preprocessed_f, "w") as wf:
            for e in tqdm(raw_examples, desc=f"Saving processed examples"):
                wf.write(e)

    def load(self, preprocessed_f: string, fields: List[Tuple[str, RawField]], **kwargs) -> List[Example]:
        with jsonlines.open(preprocessed_f, "r") as raw_examples:
            return self.load_iterable(fields, raw_examples, **kwargs)

    def load_iterable(self, fields, raw_examples, include_passage_masks=False):
        fields = list(fields.items())
        examples = []
        skip = 6
        logging.info(f"Skipping every {skip} element.")
        if len(self.langs) % skip == 0:
            raise RuntimeError("Skip value should not be a divider or multiple of number of languages!")
        for i, e in tqdm(enumerate(raw_examples), desc="Loading preprocessed data..."):
            if not self.gt_only and i % skip == 0 :
                continue
            example = self.torchtext_example(e, fields, include_passage_masks)
            examples.append(example)
            if self.data_size > 0 and self.data_size <= i:
                break
        return examples

    def torchtext_example(self, e, fields, include_passage_masks, choose_random_target=False):
        target = e["target"] if not choose_random_target else random.choice(e["target"])
        # sources = [ s[:300-1] + [self.tokenizer.eos_token_id] for s in sample(e["sources"], 30) ]
        _preprocessed_example = [
            e["id"],
            e["question"],
            e["answers"],
            e["lang"],
            e['sources'],
            [[1] * len(x) for x in e['sources']],
            e.get("doc_masks", None),
            target[0],
            [1] * len(target[0])]
        if not include_passage_masks:
            del _preprocessed_example[-3]
        example = Example.fromlist(_preprocessed_example, fields)
        return example

    def get_example_list(self):

        examples = []
        if self.preprocess:
            logging.info("Preprocessing data from MKQA...")
            raise NotImplementedError("Use preprocessor directly and pass the preprocessed data. "
                                      "This has been removed to get rid of Lucene dependency for training.")
            # preprocessor = MKQAPrep(self.langs, topk=20)
            # preprocessed = preprocessor.preprocess(write=True)
            # logging.info("Processing samples...")
            # for idx, sample in tqdm(enumerate(preprocessed), desc="Processing samples",
            #         total=len(preprocessed)):  # TODO: parallelize?
            #     if self.is_training:
            #         examples += self.process_sample(sample)
            #     else:
            #         # Do not use same question with multiple answers in validation
            #         examples += [self.process_sample(sample)[0]]

            #     if idx == 0:
            #         logging.info("Example of input formats:")
            #         src_example1 = " ".join(self.tokenizer.convert_ids_to_tokens(examples[0]["sources"][0]))
            #         src_example2 = " ".join(self.tokenizer.convert_ids_to_tokens(examples[0]["sources"][1]))
            #         logging.info("inputs 1:")
            #         logging.info(src_example1)
            #         logging.info("inputs 2:")
            #         logging.info(src_example2)
            #         if len(examples[0]["target"]) > 1:
            #             possible_target = examples[0]["target"]
            #             if type(possible_target) == list:
            #                 possible_target = possible_target[0]
            #             target_example = " ".join(self.tokenizer.convert_ids_to_tokens(possible_target))
            #             logging.info("target:")
            #              logging.info(target_example)
        else:
            with open(self.datafile, encoding="utf-8") as f:
                num_lines = sum(1 for _ in f)
            logging.info(f"Processing samples from {self.datafile}...")
            with jsonlines.open(self.datafile) as fp:
                for idx, sample in tqdm(enumerate(fp), desc="Processing samples", total=num_lines):  # TODO: parallelize?
                    if self.is_training:
                        examples += self.process_sample(sample)
                    else:
                        # Do not use same question with multiple answers in validation
                        examples += [self.process_sample(sample)[0]]

                    if idx < 20 and len(examples) == 1:
                        # less than 20 because len(exmples) won't be evaluated every iteration if some somples are rejected
                        logging.info("Example of input formats:")
                        src_example1 = " ".join(self.tokenizer.convert_ids_to_tokens(examples[0]["sources"][0]))
                        if len(examples[0]["sources"]) > 1:
                            src_example2 = " ".join(self.tokenizer.convert_ids_to_tokens(examples[0]["sources"][1]))
                        else:
                            src_example2 = "Only one example"
                        logging.info("inputs 1:")
                        logging.info(src_example1)
                        logging.info("inputs 2:")
                        logging.info(src_example2)
                        if len(examples[0]["target"]) > 1:
                            possible_target = examples[0]["target"]
                            if type(possible_target) == list:
                                possible_target = possible_target
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

    def assemble_target_sequences(self, answers: List, tokenizer: PreTrainedTokenizer):
        target_sequences = []
        with self.tokenizer.as_target_tokenizer():
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
                if type(tokenizer) in [Tokenizer, TokenizerFast]:
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

        if type(self.tokenizer) in [Tokenizer, TokenizerFast]:
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
        assert type(self.tokenizer) in [Tokenizer, TokenizerFast], f"Unsupported Tokenizer {type(self.tokenizer)}"
        assert len(sample['answers']) >= len(self.langs), \
                f"Number of languages in sample {len(sample['answers'])} is smaller than languages {len(self.langs)}"

        # get gt_index - index of golden passage, if available
        gt_index = None
        if "gt_index" in sample and sample["gt_index"] != -1:
            gt_index = sample["gt_index"]

        # if golden passage is not available, start with empty set of passages
        if not self.include_golden_passage:
            # unknown ground truth
            selected_ids = []
            titles = []
            titles_raw = []

            top_k_passages_tokens = []
            top_k_passages_raw = []
        elif gt_index is not None:  # otherwise, initialize with golden passage
            selected_ids = [(gt_index, 'en')]

            title, passage = self.db_multi.get_doc_text(gt_index+1, 'en', columns=["title", "passage"])

            titles = [self.tokenizer.encode(title, add_special_tokens=False)]
            titles_raw = [title]

            golden_passage = " " + passage
            top_k_passages_tokens = [self.tokenizer.encode(golden_passage, add_special_tokens=False)]
            top_k_passages_raw = [golden_passage]

            if self.only_gt_passages:

                question = sample['queries']['en']
                question_tokens = self.tokenizer.encode(question, add_special_tokens=False)

                input_sequences, document_masks = self.assemble_input_sequences(question=question_tokens,
                                                                                passages=top_k_passages_tokens,
                                                                                titles=titles)
                answers = sample['answers']['en'][:self.answer_limit]
                target_sequences = self.assemble_target_sequences(answers=answers)

                example = {
                    "id"       : sample["example_id"],
                    "question" : question,
                    "lang"     : 'en',
                    "answers"  : sample['answers']['en'],
                    "sources"  : input_sequences,
                    "doc_masks": document_masks,
                    "target"   : target_sequences,
                    }
                return [example]
        elif self.only_gt_passages:
            # return empty list if sample does not contain gt passage
            return []
        else:
            # gt passages are included but this sample does not contain one
            # and there are not only gt passages
            pass

        # take rest of the passages as top-k, if available
        lang_tally = { }
        number_of_contexts = len(self.langs) * self.context_size
        for neg_ind in sorted(sample['retrieval'], key=lambda x: x['score'], reverse=True):
            idx: int = neg_ind['id']
            lang: str = neg_ind['lang']

            if len(top_k_passages_tokens) == number_of_contexts:
                break

            # if passage is already included (e.g. gt during training)
            elif (idx, lang) in selected_ids:
                continue
            else:
                if lang in lang_tally:
                    lang_tally[lang] += 1
                else:
                    lang_tally[lang] = 1
                if lang_tally[lang] > self.context_size:
                    continue
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

        if len(top_k_passages_tokens) != number_of_contexts:
            logging.info("Not enough selected passages!")
            logging.info(f"Query: {sample['query']}")
            logging.info(f"Selected: {selected_ids}")
            found = [lang for _, lang in selected_ids]
            for lang in self.langs:
                if lang not in found:
                    logging.info(f"Retriever failed for {lang}")
            return []
        #assert len(top_k_passages_tokens) == number_of_contexts, \
        #    f"Passages: {len(top_k_passages_tokens)}, Context size: {number_of_contexts} \n{selected_ids}"

        examples = []
        for lang in self.langs:
            question = sample['queries'][lang]
            question_tokens = self.tokenizer.encode(question, add_special_tokens=False)

            input_sequences, document_masks = self.assemble_input_sequences(question=question_tokens,
                                                                            passages=top_k_passages_tokens,
                                                                            titles=titles)
            answers = sample['answers'][lang][:self.answer_limit]
            target_sequences = self.assemble_target_sequences(answers=answers, tokenizer=self.tokenizer)

            if not target_sequences:  # in test time
                example = {
                    "id"       : sample["example_id"],
                    "question" : question,
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
                for targetSequence in target_sequences:
                    # useful for debugging
                    # rev_input = " ".join(tokenizer.convert_ids_to_tokens(inputSequence))
                    # rev_target = " ".join(tokenizer.convert_ids_to_tokens(targetSequence))
                    example = {
                        "id"       : sample["example_id"],
                        "question" : question,
                        "lang"     : lang,
                        "answers"  : sample['answers'][lang],
                        "sources"  : input_sequences,
                        "doc_masks": document_masks,
                        "target"   : [targetSequence],
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
