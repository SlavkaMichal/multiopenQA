import jsonlines as jl
import json
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
MKQA_LUCENE_LANGS = {"ar", "da", "de", "es", "en", "fi", "fr", "hu", "it", "ja",
                     "ko", "nl", "no", "pl", "pt", "ru", "sv", "th", "tr"}

TRANSLATORS = {
    'opus-mt-mul-en'  : ['ca', 'es', 'os', 'eo', 'ro', 'fy', 'cy', 'is', 'lb', 'su', 'an', 'sq', 'fr', 'ht', 'rm', 'cv',
                         'ig', 'am', 'eu', 'tr', 'ps', 'af', 'ny', 'ch', 'uk', 'sl', 'lt', 'tk', 'sg', 'ar', 'lg', 'bg',
                         'be', 'ka', 'gd', 'ja', 'si', 'br', 'mh', 'km', 'th', 'ty', 'rw', 'te', 'mk', 'or', 'wo', 'kl',
                         'mr', 'ru', 'yo', 'hu', 'fo', 'zh', 'ti', 'co', 'ee', 'oc', 'sn', 'mt', 'ts', 'pl', 'gl', 'nb',
                         'bn', 'tt', 'bo', 'lo', 'id', 'gn', 'nv', 'hy', 'kn', 'to', 'io', 'so', 'vi', 'da', 'fj', 'gv',
                         'sm', 'nl', 'mi', 'pt', 'hi', 'se', 'as', 'ta', 'et', 'kw', 'ga', 'sv', 'ln', 'na', 'mn', 'gu',
                         'wa', 'lv', 'jv', 'el', 'my', 'ba', 'it', 'hr', 'ur', 'ce', 'nn', 'fi', 'mg', 'rn', 'xh', 'ab',
                         'de', 'cs', 'he', 'zu', 'yi', 'ml', 'mul', 'en'],
    'tgt_constituents': {
        'sjn_Latn',
        'cat',
        'nan',
        'spa', 'ile_Latn', 'pap', 'mwl', 'uzb_Latn', 'mww', 'hil', 'lij', 'avk_Latn', 'lad_Latn', 'lat_Latn',
        'bos_Latn', 'oss', 'epo', 'ron', 'fry', 'cym', 'toi_Latn', 'awa', 'swg', 'zsm_Latn', 'zho_Hant', 'gcf_Latn',
        'uzb_Cyrl', 'isl', 'lfn_Latn', 'shs_Latn', 'nov_Latn', 'bho', 'ltz', 'lzh', 'kur_Latn', 'sun', 'arg',
        'pes_Thaa', 'sqi', 'uig_Arab', 'csb_Latn', 'fra', 'hat', 'liv_Latn', 'non_Latn', 'sco', 'cmn_Hans', 'pnb',
        'roh', 'chv', 'ibo', 'bul_Latn', 'amh', 'lfn_Cyrl', 'eus', 'fkv_Latn', 'tur', 'pus', 'afr', 'brx_Latn', 'nya',
        'acm', 'ota_Latn', 'cha', 'ukr', 'xal', 'slv', 'lit', 'zho_Hans', 'tmw_Latn', 'kjh', 'ota_Arab', 'war', 'tuk',
        'sag', 'myv', 'hsb', 'lzh_Hans', 'ara', 'tly_Latn', 'lug', 'brx', 'bul', 'bel', 'vol_Latn', 'kat', 'gan',
        'got_Goth', 'vro', 'ext', 'afh_Latn', 'gla', 'jpn', 'udm', 'mai', 'ary', 'sin', 'tvl', 'hif_Latn', 'cjy_Hant',
        'bre', 'ceb', 'mah', 'nob_Hebr', 'crh_Latn', 'prg_Latn', 'khm', 'ang_Latn', 'tha', 'tah', 'tzl', 'aln', 'kin',
        'tel', 'ady', 'mkd', 'ori', 'wol', 'aze_Latn', 'jbo', 'niu', 'kal', 'mar', 'vie_Hani', 'arz', 'yue', 'kha',
        'san_Deva', 'jbo_Latn', 'gos', 'hau_Latn', 'rus', 'quc', 'cmn', 'yor', 'hun', 'uig_Cyrl', 'fao', 'mnw', 'zho',
        'orv_Cyrl', 'iba', 'bel_Latn', 'tir', 'afb', 'crh', 'mic', 'cos', 'swh', 'sah', 'krl', 'ewe', 'apc', 'zza',
        'chr', 'grc_Grek', 'tpw_Latn', 'oci', 'mfe', 'sna', 'kir_Cyrl', 'tat_Latn', 'gom', 'ido_Latn', 'sgs', 'pau',
        'tgk_Cyrl', 'nog', 'mlt', 'pdc', 'tso', 'srp_Cyrl', 'pol', 'ast', 'glg', 'pms', 'fuc', 'nob', 'qya', 'ben',
        'tat', 'kab', 'min', 'srp_Latn', 'wuu', 'dtp', 'jbo_Cyrl', 'tet', 'bod', 'yue_Hans', 'zlm_Latn', 'lao', 'ind',
        'grn', 'nav', 'kaz_Cyrl', 'rom', 'hye', 'kan', 'ton', 'ido', 'mhr', 'scn', 'som', 'rif_Latn', 'vie', 'enm_Latn',
        'lmo', 'npi', 'pes', 'dan', 'fij', 'ina_Latn', 'cjy_Hans', 'jdt_Cyrl', 'gsw', 'glv', 'khm_Latn', 'smo', 'umb',
        'sma', 'gil', 'nld', 'snd_Arab', 'arq', 'mri', 'kur_Arab', 'por', 'hin', 'shy_Latn', 'sme', 'rap', 'tyv', 'dsb',
        'moh', 'asm', 'lad', 'yue_Hant', 'kpv', 'tam', 'est', 'frm_Latn', 'hoc_Latn', 'bam_Latn', 'kek_Latn', 'ksh',
        'tlh_Latn', 'ltg', 'pan_Guru', 'hnj_Latn', 'cor', 'gle', 'swe', 'lin', 'qya_Latn', 'kum', 'mad', 'cmn_Hant',
        'fuv', 'nau', 'mon', 'akl_Latn', 'guj', 'kaz_Latn', 'wln', 'tuk_Latn', 'jav_Java', 'lav', 'jav', 'ell', 'frr',
        'mya', 'bak', 'rue', 'ita', 'hrv', 'izh', 'ilo', 'dws_Latn', 'urd', 'stq', 'tat_Arab', 'haw', 'che', 'pag',
        'nno', 'fin', 'mlg', 'ppl_Latn', 'run', 'xho', 'abk', 'deu', 'hoc', 'lkt', 'lld_Latn', 'tzl_Latn', 'mdf',
        'ike_Latn', 'ces', 'ldn_Latn', 'egl', 'heb', 'vec', 'zul', 'max_Latn', 'pes_Latn', 'yid', 'mal', 'nds'}
    }

MLQA_LANGS = {"ar", "de", "en", "es", "hi", "vi", "zh"}

MLQA_PATH = "data/mlqa/MLQA_V1"


class MLQAPrep:
    def __init__(self,
                 lang_idx: Union[List[str], Dict[str, AnyStr]],
                 topk=20,
                 mkqa_path=MLQA_PATH,
                 with_ctx=False,
                 search_with_title=False):

        self.mkqa_path = mkqa_path
        self.search_with_title = search_with_title
        self.langs = [lang for lang in lang_idx]
        self.indexes = lang_idx
        self.with_ctx = with_ctx
        self.topk = topk

        if type(lang_idx) == list:
            self.indexes = {}
            for lang in lang_idx:
                self.indexes[lang] = Retriever.get_index_name(lang=lang)
        self.dev_files = []
        self.test_files = []

        for lang_context in self.langs:
            for lang_question in self.langs:
                name = f"context-{lang_context}-question{lang_question}"
                self.dev_files.append("dev/dev-" + name + ".json")
                self.test_files.append("test/test-" + name + ".json")
                pass

    def get_data_name(self, prefix):
        name = f"mlqa_{prefix}"
        name += f"_topk{self.topk}"
        name += "with_ctx" if self.with_ctx else ""
        name += "search_with_title" if self.search_with_title else ""
        if self.langs:
            for lang in self.langs:
                name += f"_{lang}"
        else:
            raise ValueError("If spacy_only is False language list must be specified!")

        return os.path.join('data/mlqa', name + '.jsonl')

    def preprocess(self, data_file=None) -> List[Dict]:
        # crate searcher
        self.searcher = Searcher()
        for lang in self.langs:
            # add indexes
            self.searcher.addLang(lang, index_dir=self.indexes[lang])
            logging.info(f"Lang: {lang}, Index directory: {self.searcher.get_index_dir(lang)}")

        samples = []

        i = 1
        num_files = len(self.dev_files) * 2
        data_file = data_file if data_file is not None else self.get_data_name("dev")
        logging.info(f"Saving dev set into {data_file}...")
        for file in self.dev_files:
            desc = f"Processing {os.path.basename(file)} ({i}/{num_files})"
            samples += self.process_file(file, desc)
            i += 1
        with jl.open(data_file, mode='w') as writer:
            for sample in samples:
                pass

    def process_file(self, writer, file, desc):
        with open(file) as fp:
            data = json.load(fp)['data']

        samples = []
        with tqdm(total=len(data), desc=desc) as pbar:
            processed = 0
            for i, mkqa_sample in enumerate(data):
                sample = {
                    'query'     : mkqa_sample['query'],
                    'queries'   : {},
                    'answers'   : {},
                    'example_id': mkqa_sample['example_id'],
                    'retrieval' : []
                    }
                # add english query to the rest
                for lang in self.langs:
                    answers = mkqa_sample['answers'][lang]
                    sample['answers'][lang] = [answer['text'] for answer in answers]
                    sample['answers'][lang] += [alias for answer in answers if 'aliases' in answer for alias in
                                                answer['aliases']]
                    sample['queries'][lang] = mkqa_sample['queries'][lang] if lang != 'en' else mkqa_sample['query']

                title = ""
                sample['gt_index'] = ""
                sample['hard_negative_ctx'] = ""

                for lang, query in sample['queries'].items():
                    docs = self.searcher.query(query + title, lang, self.topk, field='context_title')
                    sample['retrieval'] += [{'score': doc.score, 'lang': lang, 'id': doc.id} for doc in docs]
                processed += 1
                samples.append(sample)
                writer.write(sample)
                pbar.update()
        logging.info("Finished!")
        logging.info(f"Positive ctx from dpr mapping found in {found_in_dpr_map}/{processed} samples.")
        logging.info(f"Skipped {skipping}/{total} samples.")
        if write:
            writer.close()
        return samples
