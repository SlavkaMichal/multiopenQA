import os
import random
import string
import time

from click import confirm
import jsonlines
import torchtext.data
import tqdm
from transformers import PreTrainedTokenizer
from transformers import MT5Tokenizer as Tokenizer
from transformers import MT5TokenizerFast as TokenizerFast
from typing import List, Tuple, Dict, AnyStr, Optional
from moqa.db import PassageDB
from moqa.common import config
import ipdb
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
        data = MT5Dataset(
            datafile='data/preprocessed/TODO.jsonl',
            preprocess=False,  # preprocess original dataset
            model_name='google/mt5-small',
            tokenizer=tokenizer,
            db_multi=db,  # database with passages
            langs=['en', 'it', 'de'],  # languages in an example
            max_context_size=25,  # maximal amount of contexts per sample
            interactive=True,
            multi_lingual_query=True,  # use multiple languages per question
            translated_query=False,  # use translated questions
            include_golden_passage=True,  # if true one passage containing answer string will be added if found
            use_dpr_golden=True,  # if available use dpr for golden passage if include_golden_passage is true
            only_gt_passages=False,  # make sure that all passages contain answer
            examples_per_sample=5,  # creates multiple version of a sample but in different languages
            max_len=270,  # tokenized input truncation
            data_size=10,  # if lower than zero than does nothing
            is_training=True,  # does not tokenize answers
            preprocessing_truncation="truncate_only_passages",  # truncation strategy
            include_passage_masks=False,  # unnecessary
            use_cache=True,  # use cached examples
            cached_data_path=None,  #
            init_examples=True,  # if false only class will be created and data won't be loaded
            cache_dir='data/cache/generative'
            )
        data_iter = torchtext.data.Iterator(data,
                                            shuffle=True,
                                            sort=False,
                                            batch_size=1,
                                            train=False,
                                            repeat=False,
                                            device='cpu')
        for batch in data_iter:
            print("Batch:")
            print(batch)


class MT5Dataset(torchtext.data.Dataset):
    def __init__(self,
                 datafile: AnyStr,
                 preprocess: bool,  # preprocess original dataset
                 model_name: str,
                 tokenizer: PreTrainedTokenizer,
                 db_multi: Optional[PassageDB],  # database with passages
                 langs: List[str],  # languages in an example
                 max_context_size: int,  # maximal amount of contexts per sample
                 interactive: bool,
                 # if true, program will stop execution on multiple places and confirmation will be required to continue
                 multi_lingual_query: bool,  # use multiple languages per question
                 translated_query: bool,  # use translated questions
                 include_golden_passage: bool,  # if true one passage containing answer string will be added if found
                 use_dpr_golden: bool,  # if available use dpr for golden passage if include_golden_passage is true
                 only_gt_passages: bool,  # make sure that all passages contain answer
                 examples_per_sample: int,  # creates multiple version of a sample but in different languages
                 max_len: Optional[int] = None,  # tokenized input truncation
                 data_size: int = -1,
                 # if lower than zero than does nothing otherwise limits number of processed samples for testing
                 is_training: bool = True,  # does not tokenize answers
                 preprocessing_truncation="truncate_only_passages",  # truncation strategy
                 include_passage_masks: bool = False,  # unnecessary
                 use_cache=True,  # use cached examples
                 cached_data_path=None,  #
                 init_examples=True,  # if false only class will be created and data won't be loaded
                 cache_dir='data/cache/generative', **kwargs):

        self.langs = sorted(langs)
        self.cache_dir = cache_dir
        self.datafile: AnyStr = datafile
        self.model_name = model_name
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.db_multi = db_multi
        self.max_len = tokenizer.model_max_length if max_len is None else max_len
        self.is_training = is_training
        self.use_cache = use_cache
        self.cached_data_path = cached_data_path
        self.multi_lingual_query = multi_lingual_query
        self.translated_query = translated_query
        self.include_golden_passage = include_golden_passage
        self.use_dpr_golden = use_dpr_golden
        self.only_gt_passages = only_gt_passages
        self.examples_per_sample = examples_per_sample
        self.include_passage_masks = include_passage_masks
        self.preprocessing_truncation = preprocessing_truncation
        self.data_size = data_size
        self.interactive = interactive

        logging.info(f"Max number of contexts: {max_context_size}")
        self.context_size = max_context_size // len(self.langs)
        logging.info(f"Number of contexts per language: {self.context_size}")
        logging.info(f"Languages: {self.langs}")

        fields: Dict[str, torchtext.data.Field] = self.prepare_fields(tokenizer.pad_token_id)
        if not include_passage_masks and 'doc_mask' in fields:
            del fields['doc_mask']
        self.fields = fields

        if init_examples:
            if use_cache:
                preprocessed_f = self.create_preprocessed_name()
                logging.info(f"Cache file: {preprocessed_f}")
                if not os.path.exists(preprocessed_f):
                    logging.info(f"{preprocessed_f} not found! Creating new...")
                    if self.interactive and not confirm("Continue?", default=True):
                        raise KeyboardInterrupt

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
        if self.cached_data_path is not None:
            return self.cached_data_path
        without_psg_suffix = f"_withoutpassages" if not self.include_golden_passage else ""
        multi_lingual = f"_multilingual" if self.multi_lingual_query else "_monolingual"
        translated = f"_translated" if self.translated_query else "_mkqa_translations"
        with_psg_masks = "_with_passage_masks" if self.include_passage_masks else ""
        model_name = f"_{self.model_name}" if self.model_name else ""
        gt_only = "_gt_only" if self.only_gt_passages else ""
        preprocessed_f_noext = os.path.join(self.cache_dir, os.path.basename(
            self.datafile)) + f"_" \
                              f"_C{self.context_size}" \
                              f"{with_psg_masks}" \
                              f"{gt_only}" \
                              f"{multi_lingual}" \
                              f"{translated}" \
                              f"{without_psg_suffix}" \
                              f"_{self.preprocessing_truncation}" \
                              f"{model_name}"
        preprocessed_f = preprocessed_f_noext + ".jsonl"
        return preprocessed_f

    @staticmethod
    def save(preprocessed_f: string, raw_examples: List[Dict]):
        with jsonlines.jsonlines.open(preprocessed_f, "w") as wf:
            for e in tqdm.tqdm(raw_examples, desc=f"Saving processed examples"):
                wf.write(e)

    def load(self,
             preprocessed_f: string,
             fields: Dict[str, torchtext.data.RawField],
             **kwargs) -> List[torchtext.data.Example]:
        with jsonlines.jsonlines.open(preprocessed_f, "r") as raw_examples:
            return self.load_iterable(fields, raw_examples, **kwargs)

    def load_iterable(self, fields, raw_examples, include_passage_masks=False):
        fields = list(fields.items())
        examples = []
        for i, e in tqdm.tqdm(enumerate(raw_examples), desc="Loading preprocessed data..."):
            example = self.torchtext_example(e, fields, include_passage_masks)
            examples.append(example)
            if 0 < self.data_size <= i:
                break
        return examples

    @staticmethod
    def torchtext_example(e, fields, include_passage_masks, choose_random_target=False):
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
        example = torchtext.data.Example.fromlist(_preprocessed_example, fields)
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
            with jsonlines.jsonlines.open(self.datafile) as fp:
                for idx, sample in tqdm.tqdm(enumerate(fp), desc="Processing samples",
                                             total=num_lines):  # TODO: parallelize?
                    if self.is_training:
                        examples += self.process_sample(sample)
                    else:
                        # Do not use same question with multiple answers in validation
                        examples += [self.process_sample(sample)[0]]

                    if idx < 4 and len(examples) == 1:
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
    def prepare_fields(pad_t) -> Dict[str, torchtext.data.Field]:
        WORD_field = torchtext.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t)
        WORD_nested_field = torchtext.data.NestedField(WORD_field)
        PAD_field = torchtext.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)
        PAD_nested_field = torchtext.data.NestedField(PAD_field)
        MASK_field = torchtext.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=1.)
        MASK_nested_field = torchtext.data.NestedField(MASK_field)
        TGT_WORD_field = torchtext.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t)
        TGT_PAD_field = torchtext.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)

        fields = {
            'id'         : torchtext.data.RawField(),
            'question'   : torchtext.data.RawField(),
            'answers'    : torchtext.data.RawField(),
            'lang'       : torchtext.data.RawField(),
            'src'        : WORD_nested_field,
            'src_mask'   : PAD_nested_field,
            'doc_mask'   : MASK_nested_field,
            'target'     : TGT_WORD_field,
            'target_mask': TGT_PAD_field,
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
            if self.multi_lingual_query:
                for question, passages_, titles_ in zip(question, passages, titles):
                    pass

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
        [
          "answers": {lang:[answer]}
          "example_id",
          "mkqa",
          "mlqa",
          "queries": {lang:{text,
                            retrieval: [{id, score}],
                            translations: {lang:text, retrieval: [{id, score}]},
                            }
                     }
        ]

        """

        assert type(self.tokenizer) in [Tokenizer, TokenizerFast], f"Unsupported Tokenizer {type(self.tokenizer)}"
        assert set(sample['answers']) >= set(self.langs), \
            f"Number of languages in sample {len(sample['answers'])} is smaller than languages {len(self.langs)}"

        answers = sample['answers']

        # if golden passage is not available, start with empty set of passages
        top_k_titles_raw = {}
        top_k_titles_tokens = {}
        selected_ids = []
        top_k_passages_raw = {}
        top_k_passages_tokens = {}

        if self.only_gt_passages:
            raise NotImplementedError("Only gt passages are not implemented!")
        if self.translated_query:
            raise NotImplementedError("Using translations is not currently supported!")

        if self.include_golden_passage:
            gt_lang = None
            gt_index = None
            gt_passage = None
            gt_title = None
            # add gt passage
            # sample only one gt passage
            langs_copy = self.langs.copy()
            random.shuffle(langs_copy)
            while gt_index is None and langs_copy:
                # searching for gt paragraph
                gt_lang = langs_copy.pop()
                if self.use_dpr_golden and \
                        gt_lang == 'en' and \
                        sample['mkqa'] is not None and \
                        sample['mkqa']['dpr_match']:
                    # if dpr mapping is available use it!
                    gt_index = sample['mkqa']['gt_index']
                    gt_title, gt_passage = self.db_multi.get_doc_text(gt_index, gt_lang, columns=["title", "passage"])
                    sample['mkqa']['title'] = gt_title
                    sample['mkqa']['passage'] = gt_passage
                    continue

                retrieval = sample['queries'][gt_lang]['retrieval']
                for document in retrieval:
                    if 'passage' not in document:
                        # add document if it is missing
                        gt_title, gt_passage = self.db_multi.get_doc_text(gt_index, gt_lang,
                                                                          columns=["title", "passage"])
                        document['passage'] = gt_passage
                        document['title'] = gt_title

                    for answer in answers[gt_lang]:
                        if answer in document['passage']:
                            gt_index = document['id']
                            break
            # add gt passage to example
            if gt_index is not None:
                # if there is a gt passage add it
                top_k_titles_tokens[gt_lang] = [self.tokenizer.encode(gt_title, add_special_tokens=False)]
                top_k_titles_raw[gt_lang] = [gt_title]

                golden_passage = " " + gt_passage
                top_k_passages_tokens[gt_lang] = [self.tokenizer.encode(golden_passage, add_special_tokens=False)]
                top_k_passages_raw[gt_lang] = [golden_passage]
                selected_ids.append((gt_index, gt_lang))

        # take rest of the passages as top-k, if available
        for lang in self.langs:
            retrieval = sample[lang]['retrieval']
            # retrieval = sorted(retrieval, key=lambda x: x['score'], reverse=True) # should be sorted
            for document in retrieval:
                if len(top_k_titles_raw[lang]) >= self.context_size:
                    # if there are enough passages continue to next language
                    break
                if (document['id'], lang) in selected_ids:
                    # check if it was not already added
                    continue
                if 'passage' not in document:
                    # add document if it is missing
                    title, passage = self.db_multi.get_doc_text(document['id'], lang, columns=["title", "passage"])
                    document['passage'] = passage
                    document['title'] = title

                selected_ids.append((document['id'], lang))
                title = document['title']
                passage = document['passage']

                # sometimes, there can be duplicate passages inside text (e.g. DPR passages), remove these
                if lang == 'en' and \
                        title in top_k_titles_raw['en'] and \
                        passage in top_k_passages_raw['en']:
                    continue

                top_k_titles_tokens[lang].append(self.tokenizer.encode(title, add_special_tokens=False))
                top_k_titles_raw[lang].append(title)

                passage = " " + passage
                tokenized_passage = self.tokenizer.encode(passage, add_special_tokens=False)
                top_k_passages_tokens[lang].append(tokenized_passage)
                top_k_passages_raw[lang].append(passage)

        for lang, passages in top_k_passages_raw.items():
            if len(passages) != self.context_size:
                logging.info("Not enough selected passages!")
                logging.info(f"Id: {sample['example_id']}")
                logging.info(f"Lang: {lang}")
                logging.info(f"Selected: {selected_ids}")
                logging.info(f"Expected: {self.context_size}")
                logging.info(f"Got: {len(passages)}")
                ipdb.set_trace()
                return []
        # assert len(top_k_passages_tokens) == number_of_contexts, \
        #    f"Passages: {len(top_k_passages_tokens)}, Context size: {number_of_contexts} \n{selected_ids}"

        examples = []
        answer_langs = random.sample(self.langs, self.examples_per_sample)
        for answer_lang in answer_langs:
            question = sample['queries'][answer_lang]['text']
            if self.multi_lingual_query:
                # question is in the same language as passage
                # answer language is indicated by the first question
                question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
                input_sequences, document_masks = self.assemble_input_sequences(question=question_tokens,
                                                                                passages=top_k_passages_tokens[
                                                                                    answer_lang],
                                                                                titles=top_k_titles_tokens[answer_lang])
                titles_tokens = top_k_titles_tokens[answer_lang]
                text = top_k_titles_raw[answer_lang]
                for lang in self.langs:
                    # this was already added
                    if lang == answer_lang:
                        continue

                    question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
                    _input_sequences, _document_masks = self.assemble_input_sequences(question=question_tokens,
                                                                                      passages=top_k_passages_tokens[
                                                                                          lang],
                                                                                      titles=top_k_titles_tokens[lang])
                    titles_tokens += top_k_titles_tokens[lang]
                    titles_raw += top_k_titles_raw[answer_lang]
                    input_sequences += _input_sequences
                    document_masks += _document_masks
            else:
                question_tokens = self.tokenizer.encode(question[answer_lang], add_special_tokens=False)
                # flatten passages in language order
                passages_tokens = sum([top_k_passages_tokens[lang] for lang in self.langs], [])
                titles_tokens = sum([top_k_titles_tokens[lang] for lang in self.langs], [])
                titles_raw = sum([top_k_titles_raw[lang] for lang in self.langs], [])
                input_sequences, document_masks = self.assemble_input_sequences(question=question_tokens,
                                                                                passages=passages_tokens,
                                                                                titles=titles_tokens)
            answer_raw = [sample['answers'][answer_lang][0]]
            target_sequences = self.assemble_target_sequences(answers=answer_raw,
                                                              tokenizer=self.tokenizer) if self.is_training else [
                self.tokenizer.pad_token_id]
            # useful for debugging
            # rev_input = " ".join(tokenizer.convert_ids_to_tokens(inputSequence))
            # rev_target = " ".join(tokenizer.convert_ids_to_tokens(targetSequence))
            example = {
                "id"       : sample["example_id"],
                "question" : question,
                "lang"     : answer_lang,
                "answers"  : sample['answers'][answer_lang],
                "sources"  : input_sequences,
                "doc_masks": document_masks,
                "target"   : target_sequences,
                }
            if not self.include_doc_masks:
                del example["doc_masks"]
            examples.append(example)
        return examples


if __name__ == "__main__":
    main()
