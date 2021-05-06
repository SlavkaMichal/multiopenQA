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
    "nl",  # opus-mt-mul-en nld
    "pl",  # opus-mt-mul-en pol
    "pt",  # opus-mt-mul-en por
    "ru",  # opus-mt-mul-en rus
    "sv",  # opus-mt-mul-en swe
    "th",  # opus-mt-mul-en tha
    "tr"  # opus-mt-mul-en tur
    ]

DATA_PATH = '/home/xslavk01/multiopen_QA/multiopenQA/'
cached_data = {
    'train': DATA_PATH + 'data/cache/data/mkqa_dpr_spacy_only.jsonl_mkqa_C1_answers_1_withoutpassages_truncate_whole_input_TRAIN'
                         '.jsonl',
    'val'  : DATA_PATH + 'data/cache/data/mkqa_dpr_spacy_only.jsonl_mkqa_C1_answers_1_withoutpassages_truncate_whole_input_VAL'
                         '.jsonl',
    'test' : DATA_PATH + 'data/cache/data/mkqa_dpr_spacy_only.jsonl_mkqa_C1_answers_1_withoutpassages_truncate_whole_input_TEST'
                         '.jsonl',
    }
preprocessed_data = {
    'train': DATA_PATH + 'data/preprocessed/mkqa_TRAIN_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    'val'  : DATA_PATH + 'data/preprocessed/mkqa_DEV_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    'test' : DATA_PATH + 'data/preprocessed/mkqa_TEST_split_topk20_ar_da_de_es_fi_fr_hu_it_ja_nl_pl_pt_ru_sv_th_tr_en.jsonl',
    # 'data/mkqa/mkqa_dpr_spacy_only.jsonl'
    }

config = {
    'interactive'                   : False,  # confiramtion checkpoints to continue
    'reader_tokenizer_type'         : 'google/mt5-small',
    'reader_transformer_type'       : 'google/mt5-small',
    'reader_max_input_length'       : None,
    'pretrained_model'              : None,
    # 'data/models/generative_reader_EM0.5339_S3406_Mgoogle_mt5-small_None_pcknot4',
    'load_optimizer_state_dict'     : False,
    #    'cache_transformers'      : '../../data/cache/Transformers',
    'cache_transformers'            : DATA_PATH + 'data/cache/Transformers',

    'fusion_strategy'               : 'allinputs',
    'preprocessing_truncation'      : 'truncate_whole_input',
    'fp16'                          : False,

    'save_dir'                      : DATA_PATH + 'data/models',
    'results'                       : DATA_PATH + 'data/results',  # set to None if results shoul not be saved
    'log_results'                   : True,
    'save_em_threshold'             : 0.1,
    'languages'                     : languages,

    'validation_batch_size'         : 1,
    'validate_after_steps'          : 500,
    'max_steps'                     : 10_000,
    'batch_size'                    : 1,
    'true_batch_size'               : 64,

    # 'data'                    : '../../data/mkqa/mkqa_dpr_spacy_only.jsonl',
    'data'                          : preprocessed_data,
    'preprocess'                    : False,
    'test_only'                     : False,
    'data_size'                     : -1,
    # limit number of examples for debugging, if lower than 0 than no limit is applied
    'multi_lingual_query'           : True,  # example has query in multiple languages not only in one
    'multi_lingual_answer_lang_code': True,
    'translated_query'              : False,  # use machine translated queries
    'include_golden_passage'        : True,  # include golden query if substring matches
    'only_gt_passages'              : False,  # use only passages containing answer string
    'use_dpr_golden'                : True,
    'examples_per_sample'           : 5,  # number of samples created from each sample
    'max_len'                       : 270,  # max context length
    'max_context_size'              : 25,  # max number of contexts

    # 'cache_data'              : '../../data/cache/data',
    'cache_data'                    : DATA_PATH + 'data/cache/data',
    'cached_data'                   : {},  # override default naming scheme
    'split_ratio'                   : [12, .3, 0.3],  # obsolete
    'split_random_state'            : 9613,  # make sure splits are the same each time
    # 'database'                : '../../data/wiki/multi_passage.db',
    'database'                      : DATA_PATH + 'data/wiki/multi_passage.db',
    'answer_limit'                  : 1,  # max number of answers per example

    'optimizer'                     : 'adam',
    # Parameters used in efficientQA
    "learning_rate"                 : 1e-4,
    "adam_eps"                      : 1e-06,
    "max_grad_norm"                 : 1.,
    "weight_decay"                  : 1e-5,
    "hidden_dropout"                : 0.1,
    "attention_dropout"             : 0.1,
    'scheduler'                     : 'linear',
    'scheduler_warmup_steps'        : -1
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
        if config['interactive']:
            print("here")
            ipdb.post_mortem()
        logging.error(be)
        logging.error(traceback.format_exc())
        raise be
