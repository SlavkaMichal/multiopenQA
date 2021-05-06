import logging
import os
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
    "ko",  #
    "nl",  # opus-mt-mul-en nld
    "no",  #
    "pl",  # opus-mt-mul-en pol
    "pt",  # opus-mt-mul-en por
    "ru",  # opus-mt-mul-en rus
    "sv",  # opus-mt-mul-en swe
    "th",  # opus-mt-mul-en tha
    "tr",  # opus-mt-mul-en tur
    ]

cached_data = {
    'train': 'data/cache/data/mkqa_dpr_spacy_only.jsonl_mkqa_C1_answers_1_withoutpassages_truncate_whole_input_TRAIN'
             '.jsonl',
    'val'  : 'data/cache/data/mkqa_dpr_spacy_only.jsonl_mkqa_C1_answers_1_withoutpassages_truncate_whole_input_VAL'
             '.jsonl',
    'test' : 'data/cache/data/mkqa_dpr_spacy_only.jsonl_mkqa_C1_answers_1_withoutpassages_truncate_whole_input_TEST'
             '.jsonl',
    }

config = {
    'reader_tokenizer_type'    : 'google/mt5-small',
    'reader_transformer_type'  : 'google/mt5-small',
    'reader_max_input_length'  : None,
    'pretrained_model'         : None, #'data/models/generative_reader_EM0.5339_S3406_Mgoogle_mt5-small_None_pcknot4',
    'load_optimizer_state_dict': False,
    #    'cache_transformers'      : '../../data/cache/Transformers',
    'cache_transformers'       : 'data/cache/Transformers',

    'fusion_strategy'          : 'allinputs',
    'include_golden_passage'   : False,
    'only_gt_passages'         : False,
    'preprocessing_truncation' : 'truncate_whole_input',
    'fp16'                     : False,

    'save_dir'                 : 'data/models',
    'results'                  : 'data/results',
    'log_results'              : True,
    'save_em_threshold'        : 0.4,
    'languages'                : languages,

    'validation_batch_size'    : 1,
    'validate_after_steps'     : 500,
    'max_steps'                : 10_000,
    'batch_size'               : 1,
    'true_batch_size'          : 64,

    # 'data'                    : '../../data/mkqa/mkqa_dpr_spacy_only.jsonl',
    'data'                     : 'data/mkqa/mkqa_dpr_spacy_only.jsonl',
    'preprocess'               : False,
    'test_only'                : False,
    'data_size'                : -1,  # limit number of examples for debugging, if lower than 0 than no limit is applied
    # 'cache_data'              : '../../data/cache/data',
    'cache_data'               : 'data/cache/data',
    'cached_data'              : cached_data,  # override default naming scheme
    'split_ratio'              : [12, .3, 0.3],
    'split_random_state'       : 9613,  # make sure splits are the same each time
    # 'database'                : '../../data/wiki/multi_passage.db',
    'database'                 : 'data/wiki/multi_passage.db',
    'context_per_language'     : 1,  # this is per language
    'max_len'                  : 270,  # max context length
    'answer_limit'             : 1,  # max number of answers per example

    'optimizer'                : 'adam',
    # Parameters used in efficientQA
    "learning_rate"            : 1e-4,
    "adam_eps"                 : 1e-06,
    "max_grad_norm"            : 1.,
    "weight_decay"             : 1e-5,
    "hidden_dropout"           : 0.1,
    "attention_dropout"        : 0.1,
    'scheduler'                : 'linear',
    'scheduler_warmup_steps'   : -1
    }

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
        ipdb.post_mortem()
        logging.error(be)
        logging.error(traceback.format_exc())
        raise be
