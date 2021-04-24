class Config:
    languages = [
        "ar",
        "da",
        "de",
        "es",
        "en",
        "fi",
        "fr",
        "hu",
        "it",
        "ja",
        "nl",
        "pl",
        "pt",
        "ru",
        "sv",
        "th",
        "tr"
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
        'log_results'                   : False,
        'save_em_threshold'             : 0.15,
        'languages'                     : languages,

        'validation_batch_size'         : 1,
        'validate_after_steps'          : 500,
        'max_steps'                     : 10_000,
        'batch_size'                    : 1,
        'true_batch_size'               : 64,

        # 'data'                    : '../../data/mkqa/mkqa_dpr_spacy_only.jsonl',
        'data'                          : preprocessed_data,
        'preprocess'                      : False,
        'test_only'                       : False,
        'data_size'                       : -1,
        # limit number of examples for debugging, if lower than 0 than no limit is applied
        'multi_lingual_query'             : False,  # example has query in multiple languages not only in one
        'multi_lingual_answer_lang_code'  : False,
        'translated_query'                : False,  # use machine translated queries
        'translated_retrieval_search'     : False,
        'english_ctxs_only'               : False,
        'include_golden_passage'          : False,  # include golden query if substring matches
        'only_gt_passages'                : False,  # use only passages containing answer string
        'use_dpr_golden'                  : False,

        # test split
        'test_translated_retrieval_search': True,
        'test_irrelevant_passage_langs'   : None,  # or list, e.g.: ['ar', 'en']
        "test_translated_query"           : True,  # use translated questions
        "test_include_golden_passage"     : False,
        "test_use_dpr_golden"             : False,
        "test_only_gt_passages"           : False,

        'examples_per_sample'             : 5,  # number of samples created from each sample
        'max_len'                         : 270,  # max context length
        'max_context_size'                : 25,  # max number of contexts

        # 'cache_data'              : '../../data/cache/data',
        'cache_data'                      : DATA_PATH + 'data/cache/data',
        'cached_data'                     : {},  # override default naming scheme
        'split_ratio'                     : [12, .3, 0.3],  # obsolete
        'split_random_state'              : 9613,  # make sure splits are the same each time
        # 'database'                : '../../data/wiki/multi_passage.db',
        'database'                        : DATA_PATH + 'data/wiki/multi_passage.db',

        'optimizer'                       : 'adam',
        # Parameters used in efficientQA
        "learning_rate"                   : 1e-4,
        "adam_eps"                        : 1e-06,
        "max_grad_norm"                   : 1.,
        "weight_decay"                    : 1e-5,
        "hidden_dropout"                  : 0.1,
        "attention_dropout"               : 0.1,
        'scheduler'                       : 'linear',
        'scheduler_warmup_steps'          : -1
        }
