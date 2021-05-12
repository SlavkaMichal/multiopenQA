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

preprocessed_data = {
    'test': Config.DATA_PATH + 'data/preprocessed/mkqa_TEST_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    }

pretrained_model = 'experiments/nq_open/generative_reader_EM0.2800_S14844_Mt5-small_21-05-07_06:10:08_pcknot3'

config = Config.config
config_changes = {
    'reader_tokenizer_type'           : 't5-small',
    'reader_transformer_type'         : 't5-small',

    "test_translated_query"           : True,  # use translated questions
    'test_translated_retrieval_search': True,

    'test_only'                       : True,
    'log_results'                     : True,
    'data'                            : preprocessed_data,
    # 'languages'                       : ['en'],
    'examples_per_sample'             : len(Config.languages),  # number of samples created from each sample
    "multi_lingual_query"             : False,
    'multi_lingual_answer_lang_code'  : False,
    'pretrained_model'                : Config.DATA_PATH + pretrained_model,
    "english_ctxs_only"               : True,
    'test_irrelevant_passage_langs'   : None,  # or list, e.g.: ['ar', 'en']
    "test_include_golden_passage"     : False,
    "test_use_dpr_golden"             : False,
    "test_only_gt_passages"           : False,
    'max_context_size'                : 17,  # max number of contexts
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
