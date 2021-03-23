import jsonlines as jl
from typing import List, Dict, AnyStr, Union
from moqa.common import config
import os
from moqa.retrieval import Searcher, Retriever
import logging
from tqdm import tqdm

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=config.log_file,
    level=config.log_level)

MKQA_PATH = "data/mkqa/mkqa.jsonl"

DPR_MAP = {'dev'  : "data/data_martin_nq/nq-open_dev_short_maxlen_5_ms_with_dpr_annotation.jsonl",
           'train': "data/data_martin_nq/nq-open_train_short_maxlen_5_ms_with_dpr_annotation.jsonl"}


def main():
    data = MKQAPrep({'da': 'data/indexes/demo.index'},
                    topk=10,
                    spacy_only=False,
                    with_nq=False,
                    with_translated_positive_ctx=False,
                    search_with_title=False,
                    dpr_map=DPR_MAP['train'],
                    mkqa_path=MKQA_PATH,
                    search_by_translated_ctx=False)
    data.preprocess(write=True, test=100)


class MKQAPrep:
    def __init__(self,
                 lang_idx: Union[List[str], Dict[str, AnyStr]],
                 topk=20,
                 mkqa_path=MKQA_PATH,
                 spacy_only=False,
                 with_nq=False,
                 with_translated_positive_ctx=False,
                 search_with_title=False,
                 dpr_map=DPR_MAP['train'],
                 search_by_translated_ctx=False):

        if with_nq:
            raise NotImplemented("This will add NQ with mappings to dpr and translations.")
        if search_by_translated_ctx:
            raise NotImplemented("Looking up contexts from other languages by translating English mapping.")
        if with_translated_positive_ctx:
            raise NotImplemented("Translate English positive context if found.")

        self.mkqa_path = mkqa_path
        self.search_by_translated_ctx = search_by_translated_ctx
        self.search_with_title = search_with_title
        self.langs = [lang for lang in lang_idx]
        self.indexes = lang_idx
        if type(lang_idx) == list:
            self.indexes = {}
            for lang in lang_idx:
                self.indexes[lang] = Retriever.get_index_name(lang=lang)
        self.topk = topk
        self.spacy_only = spacy_only
        self.with_nq = with_nq
        self.dpr_map = {}
        # map dpr by id
        with jl.open(dpr_map) as dpr_map:
            for sample in dpr_map:
                self.dpr_map[sample['example_id']] = sample

        self.data_file = self.get_data_name()

    def get_data_name(self):
        name = "mkqa_dpr"
        if self.spacy_only:
            name += "_spacy_only"
        elif self.langs:
            for lang in self.langs:
                name += f"_{lang}"
        else:
            raise ValueError("If spacy_only is False language list must be specified!")

        return os.path.join('data/mkqa', name + '.jsonl')

    def preprocess(self, write: bool = False, data_file=None, test=-1) -> List[Dict]:
        if not self.langs and self.spacy_only:
            raise NotImplementedError("Spacy only is not implemented and won't be")
            # self.langs = [info['lang'] for info in return_true('spacy', True)]

        # crate searcher
        searcher = Searcher()
        for lang in self.langs:
            # add indexes
            searcher.addLang(lang, index_dir=self.indexes[lang])
            logging.info(f"Lang: {lang}, Index directory: {searcher.get_index_dir(lang)}")

        if write:
            data_file = data_file if data_file is not None else self.data_file
            logging.info(f"Saving into {data_file}...")
            writer = jl.open(data_file, mode='w')
        else:
            logging.info(f"Not saving data!")

        samples = []
        total = 10000 if test == -1 else test
        with tqdm(total=total, desc="Preprocessing MKQA") as pbar, jl.open(self.mkqa_path) as mkqa:
            found_in_dpr_map = 0
            skipping = 0
            processed = 0
            for i, mkqa_sample in enumerate(mkqa):
                if i == test:
                    break
                unanswerable = False
                for answer in mkqa_sample['answers']['en']:
                    if answer['type'] in ['unanswerable', 'long_answer']:
                        unanswerable = True
                        break
                if unanswerable:
                    skipping += 1
                    pbar.update()
                    continue

                sample = {
                    'query'     : mkqa_sample['query'],
                    'queries'   : {},
                    'answers'   : {},
                    'example_id': mkqa_sample['example_id'],
                    'retrieval' : []
                    }
                # add english query to the rest
                # remove unnecessary fields
                # for lang, answers in mkqa_sample['answers'].items():
                for lang in self.langs:
                    answers = mkqa_sample['answers'][lang]
                    sample['answers'][lang] = [answer['text'] for answer in answers]
                    sample['answers'][lang] += [alias for answer in answers if 'aliases' in answer for alias in
                                                answer['aliases']]
                    sample['queries'][lang] = mkqa_sample['queries'][lang] if lang != 'en' else mkqa_sample['query']

                title = ""
                if mkqa_sample['example_id'] in self.dpr_map and self.dpr_map[mkqa_sample['example_id']]['is_mapped']:
                    found_in_dpr_map += 1
                    dpr_map = self.dpr_map[mkqa_sample['example_id']]
                    sample['gt_index'] = dpr_map['contexts']['positive_ctx']
                    sample['hard_negative_ctx'] = dpr_map['contexts']['hard_negative_ctx']
                    if self.search_with_title:
                        title = f" {dpr_map['title']}"

                for lang, query in sample['queries'].items():
                    docs = searcher.query(query + title, lang, self.topk, field='context_title')
                    sample['retrieval'] += [{'score': doc.score, 'lang': lang, 'id': doc.id} for doc in docs]
                processed += 1
                samples.append(sample)
                if write:
                    writer.write(sample)
                pbar.update()
        logging.info("Finished!")
        logging.info(f"Positive ctx from dpr mapping found in {found_in_dpr_map}/{processed} samples.")
        logging.info(f"Skipped {skipping}/{total} samples.")
        if write:
            writer.close()
        return samples


def test_debugger():
    data = MKQAPrep({'da': '../../data/indexes/demo.index'},
                    topk=10,
                    spacy_only=False,
                    with_nq=False,
                    with_translated_positive_ctx=False,
                    search_with_title=False,
                    dpr_map="../../" + DPR_MAP['train'],
                    mkqa_path="../../" + MKQA_PATH,
                    search_by_translated_ctx=False)
    data.preprocess(write=False, test=20)


if __name__ == "__main__":
    main()
    # test_debugger()
