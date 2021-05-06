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
    'train': Config.DATA_PATH + 'data/preprocessed/nq-open_TRAIN_topk20_en.jsonl',
    'val'  : Config.DATA_PATH + 'data/preprocessed/nq-open_DEV_topk20_en.jsonl',
    'test' : Config.DATA_PATH + 'data/preprocessed/nq-open_TEST_topk20_en.jsonl',
    }

checkpoint = Config.DATA_PATH + 'experiments/nq_open/generative_reader_EM0.2749_S9896_Mt5-small_21-04-26_15:40:54_pcknot3'
config = Config.config
config_changes = {
    'reader_tokenizer_type'           : 't5-small',
    'reader_transformer_type'         : 't5-small',
    'languages'                       : ['en'],
    'pretrained_model'                : checkpoint,
    'load_optimizer_state_dict'       : True,
    'max_steps'                       : 15_000,
    'log_results'                     : False,
    'data'                            : preprocessed_data,
    'multi_lingual_query'             : False,  # example has query in multiple languages not only in one,
    'translated_query'                : False,
    'use_dpr_golden'                  : False,
    'multi_lingual_answer_lang_code'  : False,
    'translated_retrieval_search'     : False,
    'english_ctxs_only'               : True,
    'include_golden_passage'          : False,  # include golden query if substring matches
    'only_gt_passages'                : False,  # use only passages containing answer string
    'test_translated_retrieval_search': False,
    'test_irrelevant_passage_langs'   : None,  # or list, e.g.: ['ar', 'en']
    "test_translated_query"           : False,  # use translated questions
    "test_include_golden_passage"     : False,
    "test_use_dpr_golden"             : False,
    "test_only_gt_passages"           : False,
    'max_context_size'                : 17,  # max number of contexts
    'examples_per_sample'             : 1,  # number of samples created from each sample
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
