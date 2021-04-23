import jsonlines as jl
import argparse
import numpy as np
import json
from typing import List, Dict, AnyStr, Union
from moqa.common import config
import os
from moqa.retrieval import Searcher
import logging
from tqdm import tqdm
from random import shuffle
import ipdb
from moqa.translate import Translator
from argparse import Namespace
from typing import TypedDict, Optional
import torch
import ipdb

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    # filename=config.log_file,
    level=config.log_level)


class MlqaData(TypedDict):
    gt_title: str
    gt_paragraph: str


class MkqaData(TypedDict):
    dpr_match: bool
    gt_index: Optional[int]
    hard_negative_index: Optional[int]


class TranslatedQuery(TypedDict):
    text: AnyStr
    retrieval: List[dict]


class Query(TranslatedQuery):
    translations: Optional[TranslatedQuery]


class Sample(TypedDict):
    mlqa: Optional[MlqaData]
    mkqa: Optional[MkqaData]
    queries: Dict[str, Query]
    answers: Dict[str, List[str]]
    example_id: Union[int, str]


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = Preprocessor(topk=args.topk,
                                mlqa_path=args.mlqa_path,
                                mkqa_path=args.mkqa_path,
                                nq_open_path=args.nq_open_path,
                                search_with_title=args.search_with_title,
                                langs=args.langs,
                                translate=args.translate,
                                retrieval=args.retrieval,
                                test=args.test,
                                device=device)

    try:
        preprocessor.preprocess(args.dataset, args.mkqa_split_ratio, args.save_file)
    except Exception as e:
        print(e)
        ipdb.post_mortem()


class Preprocessor:
    MKQA_LUCENE_LANGS = {"ar", "da", "de", "es", "en", "fi", "fr", "hu", "it", "ja",
                         "ko", "nl", "no", "pl", "pt", "ru", "sv", "th", "tr"}
    MLQA_LANGS = {"ar", "de", "en", "es", "hi", "vi", "zh"}
    MLQA_PATH = "data/mlqa/MLQA_V1/"
    MKQA_PATH = "data/mkqa/mkqa.jsonl"
    NQ_OPEN_PATH = "data/data_martin_nq"
    NQ_OPEN_FILES = {'dev'  : "nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl",
                     'train': "nq-open_train_short_maxlen_5_ms_with_dpr_annotation.jsonl",
                     'test' : "NQ-open-test.jsonl"}

    # ORIG -> one question has sample in each language with MKQA translations, question is repeated in original language
    # TRANSLATE_QUESTION -> each question has sample in each language with MT translations
    # ORIG_SELECT_RANDOM -> sample one example from ORIG
    # MKQA_TRANSLATION   -> Martin's suggestion, using MKQA translations each paragraph paired with respective question
    MKQA_PROCESS = Namespace(ORIG=0, TRANSLATE_QUESTION=1, ORIG_SELECT_RANDOM=3, MKQA_TRANSLATION=4)

    def __init__(self,
                 topk=20,
                 mlqa_path=MLQA_PATH,
                 mkqa_path=MKQA_PATH,
                 nq_open_path=NQ_OPEN_PATH,
                 search_with_title=False,
                 langs: str = 'opus-mul',
                 translate=True,
                 retrieval=True,
                 test=False,
                 device='cpu'):


        self.mlqa_path = mlqa_path
        self.mkqa_path = mkqa_path
        self.nq_open_path = nq_open_path
        self.search_with_title = search_with_title
        self.search_field = 'context_title' if search_with_title else 'context'
        self.topk = topk
        self.device = device
        self.translate = translate
        self.retrieval = retrieval
        self.test = test

        self.searcher = Searcher()
        self.translator = Translator(device=self.device)
        self.languages = {
            'opus-mul'   : self.translator.opus_mul_langs + ['en'],
            'mkqa-lucene': list(self.MKQA_LUCENE_LANGS),
            'mlqa'       : list(self.MLQA_LANGS),
            'mlqa-mkqa'  : list(self.MLQA_LANGS.intersection(self.MKQA_LUCENE_LANGS)),
            }
        self.langs = self.languages[langs]

        for lang in self.langs:
            # add indexes
            index_dir = self.searcher.get_index_name(lang)
            self.searcher.addLang(lang, index_dir=index_dir)
            logging.info(f"Lang: {lang}, Index directory: {index_dir}")

        # mlqa data file names
        self.mlqa_dev_files = {}
        self.mlqa_test_files = {}
        for lang_question in self.langs:
            name = f"context-{lang_question}-question{lang_question}"
            self.mlqa_dev_files['lang_question'] = "dev/dev-" + name + ".json"
            self.mlqa_test_files['lang_question'] = "test/test-" + name + ".json"

    def get_data_name(self, dataset, prefix=''):
        name = f"{dataset}_{prefix}" if prefix else f"{dataset}"
        name += f"_topk{self.topk}"
        name += "search_with_title" if self.search_with_title else ""
        if self.langs:
            for lang in self.langs:
                name += f"_{lang}"
        else:
            raise ValueError("If spacy_only is False language list must be specified!")

        path = 'data/preprocessed'
        if not os.path.isdir(path):
            os.mkdir(path)
        return os.path.join(path, name + '.jsonl')

    def save_samples(self, samples: List[Sample], data_file, dataset, split):
        data_file = data_file if data_file is not None else self.get_data_name(dataset, split)
        logging.info(f"Saving {dataset} {split} set into {data_file}")
        with open(data_file, 'w') as fp:
            writer = jl.Writer(fp)
            logging.info(f"Saving: {data_file}")
            writer.write_all(samples)
            writer.close()

    def preprocess(self, dataset, mkqa_split_ratio=None, data_file=None) -> List[Sample]:
        if self.test:
            ipdb.set_trace()
        if dataset == 'mkqa':
            samples = self.process_mkqa(mkqa_split_ratio)
            if 'train' in samples:
                self.save_samples(samples['train'], data_file, dataset, 'TRAIN')
            if 'dev' in samples:
                self.save_samples(samples['dev'], data_file, dataset, 'DEV')
            if 'test' in samples:
                self.save_samples(samples['test'], data_file, dataset, 'TEST')
            if 'no_answer' in samples:
                self.save_samples(samples['no_answer'], data_file, dataset, 'NO_ANSWER')
        elif dataset == 'mlqa':
            num_files = len(self.mlqa_dev_files) * 2
            i = 1
            samples = []
            for lang, file in self.mlqa_dev_files.items():
                desc = f"Processing {os.path.basename(file)} ({i}/{num_files})"
                samples = self.process_mlqa(lang, file, desc)
                i += 1
            self.save_samples(samples, data_file, dataset, "DEV")
            samples = []
            for lang, file in self.mlqa_dev_files.items():
                desc = f"Processing {os.path.basename(file)} ({i}/{num_files})"
                samples += self.process_mlqa(lang, file, desc)
                i += 1
            self.save_samples(samples, data_file, dataset, "TEST")
        elif dataset == 'nq-open':
            nq_data_file = os.path.join(self.nq_open_path, self.NQ_OPEN_FILES['train'])
            samples = self.process_nq_open(nq_data_file)
            self.save_samples(samples, data_file, dataset, "TRAIN")

            nq_data_file = os.path.join(self.nq_open_path, self.NQ_OPEN_FILES['dev'])
            samples = self.process_nq_open(nq_data_file)
            self.save_samples(samples, data_file, dataset, "DEV")

            nq_data_file = os.path.join(self.nq_open_path, self.NQ_OPEN_FILES['test'])
            samples = self.process_nq_open(nq_data_file)
            self.save_samples(samples, data_file, dataset, "TEST")
        else:
            raise ValueError(f"Dataset: {dataset} is not supported")

        return samples

    def process_mkqa_sample(self, mkqa_sample) -> Sample:
        mkqa_sample['queries']['en'] = mkqa_sample['query']

        mkqa_data = {
            'dpr_match'          : False,
            'gt_index'           : None,
            'hard_negative_index': None}
        if mkqa_sample['example_id'] in self.dpr_map and self.dpr_map[mkqa_sample['example_id']]['is_mapped']:
            dpr_map = self.dpr_map[mkqa_sample['example_id']]
            mkqa_data['dpr_map'] = True
            # mapping to dpr is one index off
            mkqa_data['gt_index'] = dpr_map['contexts']['positive_ctx'] + 1
            negative = dpr_map['contexts']['hard_negative_ctx']
            mkqa_data['hard_negative_index'] = negative+1 if negative is not None else None

        answers = {}
        queries = {}
        for lang in self.langs:
            answers[lang] = [answer['text'] for answer in mkqa_sample['answers'][lang]]
            # add aliases to correct answers
            answers[lang] += [alias for answer in mkqa_sample['answers'][lang]
                              if 'aliases' in answer for alias in answer['aliases']]
            queries[lang] = {'text': mkqa_sample['queries'][lang], 'retrieval': [], 'translations': {}}

        sample: Sample = {
            'mlqa'      : None,
            'mkqa'      : mkqa_data,
            'queries'   : queries,
            'answers'   : answers,
            'example_id': mkqa_sample['example_id'],
            }

        return sample

    def process_nq_open(self, file):
        # count number of samples
        total = sum(1 for _ in jl.open(file))

        with tqdm(total=total, desc="Preprocessing MKQA") as pbar, jl.open(self.mkqa_path) as data:
            samples: List[Sample] = []
            for i, nq_sample in enumerate(data):
                is_mapped = nq_sample.get('is_mapped', False)
                nq_data = {
                    'dpr_match'          : is_mapped,
                    'gt_index'           : None if not is_mapped else nq_sample['contexts']['positive_ctx'] + 1,
                    'hard_negative_index': None if not is_mapped else nq_sample['contexts']['hard_negative_ctx'] + 1
                    }

                answers = {}
                queries = {}
                if 'answer' in nq_sample:
                    answers['en'] = nq_sample['answer']
                elif 'single_span_answers' in nq_sample and nq_sample['single_span_answers']:
                    answers['en'] = nq_sample['single_span_answers']
                elif 'multi_span_answers' in nq_sample and nq_sample['multi_span_answers']:
                    answers['en'] = nq_sample['multi_span_answers']
                else:
                    ipdb.set_trace()
                    continue

                # add aliases to correct answers
                queries['en'] = {'text': nq_sample['question_text'], 'retrieval': [], 'translations': {}}

                sample: Sample = {
                    'mlqa'      : None,
                    'mkqa'      : nq_data,
                    'queries'   : queries,
                    'answers'   : answers,
                    'example_id': nq_sample['example_id'],
                    }

                samples.append(sample)
                pbar.update()
                if self.test:
                    break

        self.search_paragraphs(samples)

        return samples

    def process_mkqa(self, split_ratio=None):
        total = 10000
        split_size = 0
        if split_ratio is not None:
            split_ratio = np.array(split_ratio, dtype='f')
            split_size = split_ratio.size[0]
            if split_size != 2 or split_size != 3:
                raise ValueError(f"Split ratio must have 2 or 3 elements!")

        # map dpr by id
        self.dpr_map = {}
        nq_data_file = os.path.join(self.NQ_OPEN_PATH, self.NQ_OPEN_FILES['train'])
        with jl.open(nq_data_file) as dpr_map:
            for sample in dpr_map:
                self.dpr_map[sample['example_id']] = sample
        nq_data_file = os.path.join(self.NQ_OPEN_PATH, self.NQ_OPEN_FILES['dev'])
        with jl.open(nq_data_file) as dpr_map:
            for sample in dpr_map:
                self.dpr_map[sample['example_id']] = sample

        with tqdm(total=total, desc="Preprocessing MKQA") as pbar, jl.open(self.mkqa_path) as data:
            skipping = 0
            no_answer = []
            samples: List[Sample] = []
            for i, mkqa_sample in enumerate(data):
                unanswerable = False
                for answer in mkqa_sample['answers']['en']:
                    if answer['type'] in ['unanswerable', 'long_answer']:
                        unanswerable = True
                        break
                if unanswerable:
                    no_answer.append(mkqa_sample)
                    skipping += 1
                    pbar.update()
                    continue

                sample = self.process_mkqa_sample(mkqa_sample)
                samples.append(sample)
                pbar.update()
                if self.test:
                    break

        self.translate_samples(samples)
        self.search_paragraphs(samples)

        sample_splits = {'no_answer': no_answer}
        if split_ratio is not None:
            samples_size = len(samples)
            if split_size != 2 or split_size != 3:
                raise ValueError("Incorrect split_size, must contain 2 or 3 elements")
            split_ratio /= split_ratio.sum()
            logging.info(f"Splits ratio: {split_ratio}")
            len_train = int(samples_size * split_ratio[0])
            len_dev = int(samples_size * split_ratio[1])
            shuffle(samples)

            # flatten
            sample_splits['train'] = samples[:len_train]
            samples = samples[len_train:]
            if split_size == 2:
                sample_splits['dev'] = samples
            elif split_size == 3:
                sample_splits['dev'] = samples[:len_dev]
                sample_splits['test'] = samples[len_dev:]
        else:
            sample_splits['train'] = samples

        return sample_splits

    def process_mlqa(self, lang, file, desc):
        with open(file) as fp:
            data = json.load(fp)['data']

        samples = []
        with tqdm(total=len(data), desc=desc) as pbar:
            for i, paragraphs in enumerate(data):
                title = paragraphs['title']
                for paragraph in paragraphs['paragraphs']:
                    context = paragraph['context']
                    for qas in paragraph['qas']:
                        mlqa_data: MlqaData = {
                            'gt_title'    : title,
                            'gt_paragraph': context}
                        query: Query = {'text': qas['question'], 'retrieval': [], 'translations': {}}
                        answers = [answer['text'] for answer in qas['answers']]
                        sample: Sample = {
                            'mlqa'      : mlqa_data,
                            'mkqa'      : None,
                            'queries'   : {lang: query},
                            'answers'   : {lang: answers},  # answers in original language only
                            'example_id': qas['id'], }
                        samples.append(sample)
                pbar.update()
                if self.test:
                    break
        self.translate_samples(samples)
        self.search_paragraphs(samples)
        return samples

    def translate_samples(self, samples: List[Sample]):
        if not self.translate:
            return []
        with tqdm(total=len(samples), desc='translating...') as pbar:
            for sample in samples:
                for lang, query in sample['queries'].items():
                    text = query['text']
                    translations = self.translator.translate_opus_mul(text, lang, dst_langs=self.langs)
                    for lang, tr_query in translations.items():
                        query['translations'][lang] = {'text':translations[lang], 'retrieval': []}
                pbar.update()

        return samples

    def search_paragraphs(self, samples: List[Sample]):
        if not self.retrieval:
            return []
        with tqdm(total=len(samples), desc='searching...') as pbar:
            for sample in samples:
                for lang, query in sample['queries'].items():
                    query['retrieval'] = self.search_query(query['text'], lang)
                    for tr_lang, tr_query in query['translations'].items():
                        tr_query['retrieval'] = self.search_query(tr_query['text'], tr_lang)
                pbar.update()
        return samples

    def search_query(self, query, lang):
        docs = self.searcher.query(query, lang, self.topk, field=self.search_field)
        return [{'score': doc.score, 'id': doc.id} for doc in docs]

    def add_korean_translations(self, samples):
        pass

    def add_norwegian_translations(self, samples):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing datasets")
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--mlqa-path', type=str, default=Preprocessor.MLQA_PATH)
    parser.add_argument('--mkqa-path', type=str, default=Preprocessor.MKQA_PATH)
    parser.add_argument('--nq-open-path', type=str, default=Preprocessor.NQ_OPEN_PATH)

    parser.add_argument('--search-with-title', type=bool, default=False)
    parser.add_argument('--langs', type=str, default='opus-mul',
                        choices=['opus-mul', 'mkqa-lucene', 'mlqa', 'mlqa-mkqa'])
    parser.add_argument('--no-translation', dest='translate', action='store_false')
    parser.add_argument('--not-retrieval', dest='retrieval', action='store_false')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--dataset', type=str, default='mkqa', choices=['mkqa', 'mlqa', 'nq-open'])
    parser.add_argument('--mkqa_split_ratio', type=float, default=None, nargs='+')
    parser.add_argument('--save-file', type=str, default=None)

    args = parser.parse_args()
    main(args)
