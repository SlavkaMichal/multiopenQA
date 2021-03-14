from typing import TypedDict, Union, List, Optional


class OptimConfig(TypedDict):
    optimizer: str  # adam, adamw
    learning_rate: float
    adam_eps: float
    max_grad_norm: float
    weight_decay: float
    hidden_dropout: float
    attention_dropout: float


class SchedulerConfig(TypedDict):
    scheduler: Optional[str]  # None, linear, cosine, constant
    scheduler_warmup_steps: int
    # scheduler_training_steps: 14_400,


class TrainConfig(TypedDict):
    # loading model
    reader_tokenizer_type: str
    reader_transformer_type: str
    reader_max_input_length: Optional[int]
    pretrained_model: Optional[str]
    cache_transformers: str
    # Available fusion strategies
    # "allinputs" (considers only passage embeddings in the decoder),
    # "passages" (considers only passage embeddings in the decoder)
    # strategy allinputs works slightly better (+ ~0.15 EM)
    fusion_strategy: str
    include_golden_passage: bool
    preprocessing_truncation: str  # truncate_only_passages, truncate_whole_input

    save_dir: str
    results: str
    save_em_threshold: float
    test_only: bool
    languages: List[str]

    # training parameters
    validation_batch_size: int
    validate_after_steps: int
    max_steps: int
    batch_size: int
    true_batch_size: int

    data: str
    preprocess: bool
    cache_data: str
    split_ratio: Union[float, List[float]]
    database: str
    context_length: int

    optim_cfg: OptimConfig
    sched_cfg: SchedulerConfig
