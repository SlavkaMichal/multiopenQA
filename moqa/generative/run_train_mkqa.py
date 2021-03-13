import logging
import os
import numpy as np
import traceback
from random import randint
import random
import torch

from moqa.generative import Trainer
from moqa.generative.types import TrainConfig, OptimConfig, SchedulerConfig
from moqa.common import config as logging_cfg

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=logging_cfg.log_file,
    level=logging_cfg.log_level)

languages = [
    "ar",
    "bn",
    "da",
    "de",
    "es",
    "en",
    "fi",
    "fr",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "th",
    "tr",
    ]

optimizer_config: OptimConfig = {
    'optimizer'        : 'adam',
    # Parameters used in efficientQA
    "learning_rate"    : 1e-4,
    "adam_eps"         : 1e-06,
    "max_grad_norm"    : 1.,
    "weight_decay"     : 1e-5,
    "hidden_dropout"   : 0.1,
    "attention_dropout": 0.1,
    }

scheduler_config: SchedulerConfig = {
    'scheduler'             : 'linear',
    'scheduler_warmup_steps': 600
    }

config: TrainConfig = {
    'reader_tokenizer_type'   : 'google/mt5-small',
    'reader_transformer_type' : 'google/mt5-small',
    'reader_max_input_length' : None,
    'pretrained_model'        : None,
    'cache_transformers'      : 'data/cache/Transformers',

    'fusion_strategy'         : 'allinputs',
    'include_golden_passage'  : False,
    'preprocessing_truncation': 'truncate_only_passages',

    'save_dir'                : 'data/models',
    'results'                 : 'data/results',
    'save_em_threshold'       : 0.3,
    'test_only'               : False,
    'languages'               : languages,

    'validation_batch_size'   : 1,
    'validate_after_steps'    : 500,
    'max_steps'               : 10_000,
    'batch_size'              : 1,
    'true_batch_size'         : 64,

    'data'                    : 'data/mkqa/mkqa.jsonl',
    'cache_data'              : 'data/cache/data',
    'split_ratio'             : [0.7, 0.25],
    'database'                : 'data/wiki/multi_passage.db',
    'context_length'          : 25,

    'optim_cfg'               : optimizer_config,
    'sched_cfg'               : scheduler_config
    }

if __name__ == "__main__":
    os.mkdir(config["save_dir"])
    os.mkdir(config["results"])
    os.mkdir(config["cache_data"])

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
        logging.error(be)
        logging.error(traceback.format_exc())
        raise be
    finally:
        framework.db.close()
