import logging
import numpy as np
import traceback
from random import randint
import random
import torch
import ipdb

from moqa.generative import Trainer
# from moqa.generative.config_types import TrainConfig, OptimConfig, SchedulerConfig
import os

os.environ['log_file_suffix'] = __file__.replace('.py', '')
from moqa.common import config as logging_cfg
from moqa.generative.config import Config

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=logging_cfg.log_file,
    level=logging_cfg.log_level)

preprocessed_data = {
    'train': Config.DATA_PATH + 'data/preprocessed/mkqa_TRAIN_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    'val'  : Config.DATA_PATH + 'data/preprocessed/mkqa_DEV_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    'test' : Config.DATA_PATH + 'data/preprocessed/mkqa_TEST_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    }

config = Config.config
config['log_results'] = True
config['include_golden_passage'] = True
config['save_em_threshold'] = 0.15
config['reader_tokenizer_type'] = 't5-small'
config['reader_transformer_type'] = 't5-small'
config['data'] = preprocessed_data
config['multi_lingual_query'] = False  # example has query in multiple languages not only in one
config['translated_query'] = False
config['use_dpr_golden'] = True

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
