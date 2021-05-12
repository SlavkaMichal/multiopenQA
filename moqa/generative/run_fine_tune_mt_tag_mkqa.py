import logging
import numpy as np
import traceback
from random import randint
import random
import torch
import ipdb

from moqa.generative import Trainer
from moqa.common import config as logging_cfg
from moqa.generative.config import Config

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=logging_cfg.log_file,
    level=logging_cfg.log_level)

languages = [
    "ar",  # opus-mt-mul-en ara
    "da",  # opus-mt-mul-en dan
    "de",  # opus-mt-mul-en deu
    "es",  # opus-mt-mul-en spa
    "en",  # opus-mt-mul-en
    "fi",  # opus-mt-mul-en fin
    "fr",  # opus-mt-mul-en fra
    "hu",  # opus-mt-mul-en hun
    "it",  # opus-mt-mul-en ita
    "ja",  # opus-mt-mul-en jav
    "nl",  # opus-mt-mul-en nld
    "pl",  # opus-mt-mul-en pol
    "pt",  # opus-mt-mul-en por
    "ru",  # opus-mt-mul-en rus
    "sv",  # opus-mt-mul-en swe
    "th",  # opus-mt-mul-en tha
    "tr"  # opus-mt-mul-en tur
    ]

pretrained_model = 'experiments/ml_query_lang_code/generative_reader_EM0.2500_S4212_Mgoogle_mt5-small_21-05-08_11:10:30_pcknot4'

DATA_PATH = '/home/xslavk01/multiopen_QA/multiopenQA/'
preprocessed_data = {
    'train': DATA_PATH + 'data/preprocessed/mkqa_TRAIN_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    'val'  : DATA_PATH + 'data/preprocessed/mkqa_DEV_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    'test' : DATA_PATH + 'data/preprocessed/mkqa_TEST_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    # 'data/mkqa/mkqa_dpr_spacy_only.jsonl'
    }

config = Config.config
config_changes = {
    "test_translated_query"           : True,  # use translated questions
    "translated_query"                : True,  # use translated questions
    'test_translated_retrieval_search': True,

    'data'                            : preprocessed_data,
    'examples_per_sample'             : len(Config.languages),  # number of samples created from each sample
    "multi_lingual_query"             : True,
    'multi_lingual_answer_lang_code'  : True,
    'pretrained_model'                : Config.DATA_PATH + pretrained_model,
    "english_ctxs_only"               : False,
    'test_irrelevant_passage_langs'   : None,  # or list, e.g.: ['ar', 'en']
    "test_include_golden_passage"     : False,
    "test_use_dpr_golden"             : False,
    "test_only_gt_passages"           : False,
    }
config.update(config_changes)

if __name__ == "__main__":
    # os.mkdir(config["save_dir"])
    # os.mkdir(config["results"])
    # os.mkdir(config["cache_data"])

    seed = randint(0, 10_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f"Random seed: {seed}")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    framework = Trainer(config, device)
    try:
        r = framework.fit()
    except BaseException as be:
        if config['interactive']:
            print("here")
            ipdb.post_mortem()
        logging.error(be)
        logging.error(traceback.format_exc())
        raise be
