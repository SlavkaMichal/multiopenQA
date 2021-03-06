import copy
import json
import logging
import os
import socket
import time

import torch
import torch.nn.functional as F
import transformers
from torch.nn import DataParallel
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchtext.data import Iterator
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput

from moqa.datasets.mt5datasets import MT5Dataset
from moqa.common.model_utils import count_parameters, sum_parameters, report_parameters
from moqa.common.utils import timestamp
from moqa.common.eval_utils import metric_max_over_ground_truths, exact_match_score
from moqa.generative.model import MT5QA
from moqa.db.db import PassageDB


def get_model(m):
    if type(m) == DataParallel:
        return m.module
    return m


class Trainer:
    def __init__(self, config: dict, device):
        self.config = config
        self.device = device
        self.best_em = config["save_threshold"]
        self.increase_validation_frequency = False
        # adding special tokens
        self.tokenizer = self.init_tokenizer(config)

        self.db = PassageDB(db_path=self.config['pass_database'])

    @staticmethod
    def init_tokenizer(config) -> PreTrainedTokenizer:
        """
        Creates tokenizer and add special tokens into it
        """
        reader_tokenizer = AutoTokenizer.from_pretrained(config["reader_tokenizer_type"],
                                                         cache_dir=config["transformers_cache"])
        reader_tokenizer.question_special_token = '<question>'
        reader_tokenizer.passage_special_token = '<passage>'
        reader_tokenizer.title_special_token = '<title>'
        reader_tokenizer.add_tokens(
            [reader_tokenizer.question_special_token, reader_tokenizer.passage_special_token,
             reader_tokenizer.title_special_token], special_tokens=True)
        return reader_tokenizer

    def fit(self):
        config = self.config

        logging.debug(json.dumps(config, indent=4, sort_keys=True))

        include_passage_masks = config["fusion_strategy"] == "passages"
        fields = MT5Dataset.prepare_fields(pad_t=self.tokenizer.pad_token_id)
        training_dataset = MT5Dataset

        train = training_dataset(config["train_data"], fields=fields, tokenizer=self.tokenizer,
                                 database=self.db,
                                 cache_dir=config["data_cache_dir"],
                                 context_length=config["context_length"],
                                 include_golden_passage=config["include_golden_passage_in_training"],
                                 include_passage_masks=include_passage_masks,
                                 preprocessing_truncation=config["preprocessing_truncation"],
                                 one_answer_per_question=config.get("one_question_per_epoch", False),
                                 use_only_human_answer=config.get("use_human_answer_only", False),
                                 is_training=True)

        val = MT5Dataset(config["val_data"], fields=fields, tokenizer=self.tokenizer,
                         database=self.db,
                         cache_dir=config["data_cache_dir"],
                         context_length=config["context_length"],
                         include_passage_masks=include_passage_masks,
                         preprocessing_truncation=config["preprocessing_truncation"],
                         use_only_human_answer=config.get("use_human_answer_only", False),
                         is_training=False)

        logging.info("Loading model...")
        model = torch.load(config["model"], map_location=self.device) \
            if "model" in config \
            else MT5QA.from_pretrained().to(self.device)

        test = MT5Dataset(config["test_data"], fields=fields, tokenizer=self.tokenizer,
                          database=self.db,
                          cache_dir=config["data_cache_dir"],
                          context_length=self.config["context_length"],
                          include_passage_masks=include_passage_masks,
                          preprocessing_truncation=self.config["preprocessing_truncation"],
                          is_training=False)

        logging.info(f"Training data examples:{len(train)}")
        logging.info(f"Validation data examples:{len(val)}")
        logging.info(f"Test data examples {len(test)}")

        train_iter = Iterator(train,
                              shuffle=True,
                              sort=False,  # do not sort!
                              batch_size=1,
                              train=True,
                              repeat=False,
                              device=self.device)
        val_iter = Iterator(val,
                            sort=False,
                            shuffle=False,
                            batch_size=1,
                            repeat=False, device=self.device)
        test_iter = Iterator(test,
                             sort=False, shuffle=False,
                             batch_size=1,
                             repeat=False, device=self.device)

        logging.info(f"Resizing token embeddings to length {len(self.tokenizer)}")
        model.resize_token_embeddings(len(self.tokenizer))

        logging.info(f"Model has {count_parameters(model)} trainable parameters")
        logging.info(f"Trainable parameter checksum: {sum_parameters(model)}")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

        # Init optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params"      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
                },
            {
                "params"      : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
                }, ]
        # Optimizer setup
        if config["optimizer"] == "adamw":
            optimizer = AdamW
        elif config["optimizer"] == "adam":
            optimizer = Adam
        else:
            raise ValueError("Unsupported optimizer")
        optimizer = optimizer(optimizer_grouped_parameters,
                              lr=self.config["learning_rate"],
                              eps=self.config["adam_eps"])

        if config.get("resume_checkpoint", False):
            optimizer.load_state_dict(model.optimizer_state_dict)

            # Init scheduler
        if "scheduler_warmup_steps" in self.config:
            t_total = self.config["max_steps"]
            warmup_steps = self.config["scheduler_warmup_steps"]
            scheduler = self.init_scheduler(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total,
                last_step=get_model(model).training_steps - 1
                )
            logging.info(f"Scheduler: warmup steps: {warmup_steps}, total_steps: {t_total}")
        else:
            scheduler = None

        start_time = time.time()
        try:
            it = 0
            while get_model(model).training_steps < self.config["max_steps"]:
                logging.info(f"Epoch {it}")
                train_loss = self.train_epoch(model=model,
                                              data_iter=train_iter,
                                              val_iter=val_iter,
                                              optimizer=optimizer,
                                              scheduler=scheduler)
                logging.info(f"Training loss: {train_loss:.5f}")
                it += 1
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
            if hasattr(self, "best_ckpt_name"):
                logging.info(f"Loading best checkpoint {self.best_ckpt_name}")
                model = torch.load(self.best_ckpt_name, map_location=self.device)
        logging.info("#" * 50)
        logging.info("Validating on the test data")
        self.validate(model, test_iter)

        # best_em = 0
        # result = {"em": best_em}
        # logging.info("Fit achieved results\n" + str(result))
        # return result

    def init_scheduler(self, optimizer: Optimizer, num_warmup_steps: int,
                       num_training_steps: int, last_step: int = -1) -> LambdaLR:
        """
        Initialization of lr scheduler.

        :param optimizer: The optimizer that is used for the training.
        :type optimizer: Optimizer
        :param num_warmup_steps: Number of warm up steps.
        :type num_warmup_steps: int
        :param  num_training_steps: Number of training steps
        :type   num_training_steps: int
        :param  last_step: Last step
        :type   last_step: int
        :return: Created scheduler.
        :rtype: LambdaLR
        """
        if last_step > 0:

            # We need initial_lr, because scheduler demands it.
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])

        if self.config["scheduler"] == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=last_step)
        elif self.config["scheduler"] == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=0.5,
                last_epoch=last_step)
        elif self.config["scheduler"] == "constant":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                last_epoch=last_step)
        else:
            scheduler = None

        return scheduler

    def train_epoch(self,
                    model: MT5QA,
                    data_iter: Iterator,
                    val_iter: Iterator,
                    optimizer: Optimizer,
                    scheduler: LambdaLR):
        #  Training flags
        model.train()
        # Make sure parameters are zero
        optimizer.zero_grad()

        # Determine update ratio, e.g. if true_batch_size = 32 and batch_size=8, then
        # gradients should be updated  every 4th iteration (except for last update!)
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        assert self.config["true_batch_size"] % self.config["batch_size"] == 0
        updated = False
        adjusted_for_last_update = False  # In last update, the ba tch size is adjusted to whats left

        # Calculate total number of updates per epoch
        total = len(data_iter.data()) // data_iter.batch_size + 1

        it = tqdm(enumerate(data_iter), total=total)

        # For progressive  training loss  reporting
        total_losses = []
        losses_per_update = []

        # If we use fp16, gradients must be scaled
        grad_scaler = None
        if self.config["fp16"]:
            # noinspection PyUnresolvedReferences
            grad_scaler = torch.cuda.amp.GradScaler()

        for i, batch in it:
            assert len(batch.src) == 1  # more  than 1 example per batch is unsupported

            batch.src = batch.src[0]
            batch.src_mask = batch.src_mask[0]
            batch.doc_mask = batch.doc_mask[0] if hasattr(batch, "doc_mask") else None

            assert self.tokenizer.pad_token_id not in batch.src[batch.src_mask.bool()].view(-1).tolist()

            src_shapes = batch.src.shape
            src_mask_shapes = batch.src_mask.shape
            target_shapes = batch.target.shape
            target_mask_shapes = batch.target_mask.shape
            try:
                # Useful for debugging
                # inputs = [" ".join(self.tokenizer.convert_ids_to_tokens(inp)) for inp in batch.src]
                # passage_inputs = [" ".join(self.tokenizer.convert_ids_to_tokens(inp[doc_mask.bool()])) for inp,
                # doc_mask in zip(batch.src,batch.doc_mask)]
                # target =[" ".join(self.tokenizer.convert_ids_to_tokens(target)) for target in batch.target]

                if self.config["fp16"]:
                    # noinspection PyUnresolvedReferences
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=batch.src, attention_mask=batch.src_mask,
                                        passage_mask=batch.doc_mask,
                                        decoder_input_ids=batch.target[:, :-1].contiguous(),
                                        decoder_attention_mask=batch.target_mask[:, :-1].contiguous(),
                                        use_cache=False)
                        lm_logits = outputs[0]
                        labels = batch.target[:, 1:].reshape(-1)

                        # Adjust update ratio for last update if needed
                        if (total - i) < update_ratio and not adjusted_for_last_update:
                            update_ratio = (total - i)
                            adjusted_for_last_update = True

                        # Compute loss
                        losses = F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                                 reduction='none')
                        loss = losses.mean()
                        loss /= update_ratio
                    grad_scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids=batch.src, attention_mask=batch.src_mask,
                                    passage_mask=batch.doc_mask,
                                    decoder_input_ids=batch.target[:, :-1].contiguous(),
                                    decoder_attention_mask=batch.target_mask[:, :-1].contiguous(),
                                    use_cache=False)
                    lm_logits = outputs[0]
                    labels = batch.target[:, 1:].reshape(-1)

                    # Adjust update ratio for last update if needed
                    if (total - i) < update_ratio and not adjusted_for_last_update:
                        update_ratio = (total - i)
                        adjusted_for_last_update = True

                    losses = F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                             reduction='none')
                    loss = losses.mean()
                    loss /= update_ratio
                    loss.backward()

                # record losses to list
                losses_per_update.append(loss.item())
                if len(losses_per_update) == update_ratio and not adjusted_for_last_update:
                    # grad clipping should be applied to unscaled gradients
                    if self.config["fp16"]:
                        grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                                   self.config["max_grad_norm"])
                    # compute training loss
                    loss_per_update = sum(losses_per_update) / len(losses_per_update)
                    total_losses += losses_per_update
                    losses_per_update = []

                    if self.config["fp16"]:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    get_model(model).training_steps += 1
                    if scheduler is not None:
                        scheduler.step()
                    updated = True
                    # If we are past 2/3 of expected training steps
                    if get_model(model).training_steps > (2 * self.config["max_steps"] / 3) and \
                            not self.increase_validation_frequency:
                        # Increase validation frequency with factor of 2
                        self.config["validate_after_steps"] = self.config[
                                                                  "validate_after_steps"] // 2
                        self.increase_validation_frequency = True
                        logging.info(f"Validation frequency increased to: {self.config['validate_after_steps']}")

                    it.set_description(f"Steps: {get_model(model).training_steps}, Training loss: {loss_per_update}")

                    # Validate after every validate_after_steps steps
                    if get_model(model).training_steps > 1 and \
                       get_model(model).training_steps % self.config["validate_after_steps"] == 0:
                        self.validate(model, val_iter, optimizer_dict=optimizer.state_dict())
            # Catch out-of-memory errors
            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    torch.cuda.empty_cache()
                    logging.error("OOM detected, emptying cache...")
                    logging.error(f"src_shape {src_shapes}\n"
                                  f"src_mask_shape{src_mask_shapes}\n"
                                  f"target_shapes{target_shapes}\n"
                                  f"target_mask_shapes{target_mask_shapes}\n"
                                  )
                    time.sleep(3)
                else:
                    raise e
        if not updated:
            # Do the last step if needed
            if self.config["fp16"]:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                           self.config["max_grad_norm"])
            if self.config["fp16"]:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            get_model(model).training_steps += 1
            if scheduler is not None:
                scheduler.step()
            updated = True

        # Validate after epoch
        self.validate(model, val_iter)
        return sum(total_losses) / len(total_losses)

    @torch.no_grad()
    def validate(self, model: MT5QA, val_iter: Iterator, optimizer_dict=None):
        """
        Does not compute validation loss for now
        """
        model = model.eval()
        it = tqdm(enumerate(val_iter), total=len(val_iter.data()) // val_iter.batch_size + 1)

        total = 0
        hits = 0
        loss_list = []
        for _, batch in it:
            batch.src = batch.src[0]
            batch.src_mask = batch.src_mask[0]
            batch.doc_mask = batch.doc_mask[0] if hasattr(batch, "doc_mask") else None

            total += len(batch)
            concatenated_encoder_output, concatenated_encoder_attention = model(input_ids=batch.src,
                                                                                attention_mask=batch.src_mask,
                                                                                encode_only=True)
            concatenated_encoder_output_copy = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=copy.deepcopy(concatenated_encoder_output['last_hidden_state']))
            concatenated_encoder_attention_copy = copy.deepcopy(concatenated_encoder_attention)
            outputs: Seq2SeqLMOutput = model(input_ids=None,
                                             attention_mask=concatenated_encoder_attention_copy,
                                             encoder_outputs=concatenated_encoder_output_copy,
                                             passage_mask=batch.doc_mask,
                                             decoder_input_ids=batch.target[:, :-1].contiguous(),
                                             decoder_attention_mask=batch.target_mask[:, :-1].contiguous())

            lm_logits = outputs.logits
            labels = batch.target[:, 1:].reshape(-1)

            losses = F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                     reduction='none')
            loss_list += losses.tolist()

            # hacky, provide just some tensor as input ids, such that it matches batch dimension 1,
            # do not provide input ids, as they should not be needed (and have pre-concatenation batch dim)
            tokenized_answers = get_model(model).generate(input_ids=concatenated_encoder_attention,
                                                          # num_beams=5,
                                                          # num_return_sequences=5,
                                                          attention_mask=concatenated_encoder_attention,
                                                          encoder_outputs=concatenated_encoder_output,
                                                          decoder_start_token_id=batch.target[0][0])

            predicted_answers = [self.tokenizer.decode(ans, skip_special_tokens=True) for ans in
                                 tokenized_answers]
            for i in range(len(batch)):
                hit = metric_max_over_ground_truths(
                    metric_fn=exact_match_score, prediction=predicted_answers[i],
                    ground_truths=batch.answers[i])
                hits += int(hit)
            it.set_description(f"Val Loss: {sum(loss_list) / len(loss_list):.3f} EM: {hits / total:.3f}")

        EM = hits / total
        logging.info(f"S: {get_model(model).training_steps} Validation Loss: {sum(loss_list) / len(loss_list)}")
        logging.info(f"Validation EM: {EM}")
        if EM > self.best_em and not self.config['test_only']:
            logging.info(f"{EM} ---> New BEST!")
            self.best_em = EM
            serializable_model_name = self.config['reader_transformer_type'].replace("/", "_")
            saveable_model = get_model(model)
            saveable_model.optimizer_state_dict = optimizer_dict
            # Note that model training is fully resumable
            # it contains .optimizer_state_dict and .training_steps (=number of updates)
            saved_name = os.path.join(self.config['save_dir'], f"generative_reader_"
                                                               f"EM{EM:.4f}_"
                                                               f"S{get_model(model).training_steps}_"
                                                               f"M{serializable_model_name}_"
                                                               f"{timestamp()}_{socket.gethostname()}")
            self.best_ckpt_name = saved_name
            torch.save(saveable_model, saved_name)
            model = model.train()
        return EM


sample_cfg = {
    "reader_tokenizer_type"             : "google/mt5-small",
    "reader_transformer_type"           : "google/mt5-small",
    "reader_max_input_length"           : 250,
    # Available fusion strategies
    # "allinputs" (considers only passage embeddings in the decoder),
    # "passages" (considers only passage embeddings in the decoder)
    # strategy allinputs works slightly better (+ ~0.15 EM)
    "fusion_strategy"                   : "allinputs",

    "save_dir"                          : ".saved",  # where the checkpoints will be saved
    "results"                           : ".results",  # where validation results will be saved

    "test_only"                         : True,
    "validation_batch_size"             : 1,
    "validate_after_steps"              : 500,

    # "pre_initialize": True,
    # "pretrained_reader_model": ".saved/generative_reader_EM0.4294_S3555_Mt5-base_2020-12-25_00:42_supergpu7.fit
    # .vutbr.cz",
    ###############################
    # Data
    ###############################
    "train_data"                        : "data/reader/ranked/NQ-open_TRAINING_maxlen_5_ms_with_dpr_annotation"
                                          ".jsonl_dpr_official_nqsingle_of_impossible.jsonl",
    "val_data"                          : "data/reader/ranked/NQ-open_DEV_maxlen_5_ms_with_dpr_annotation"
                                          ".json_dpr_official_nqsingle_of_impossible.jsonl",
    "test_data"                         :
        "data/reader/ranked/NQ-open_TEST.jsonl_nq-open_dpr_official_nqsingle_of_impossible.jsonl",
    "pass_database"                     : "data/db/wiki2018_dpr_blocks.db",  # database of passages and titles
    ###############################
    # Optimization hyper-parameters
    ###############################
    # Parameters used in efficientQA
    "learning_rate"                     : 1e-4,
    "adam_eps"                          : 1e-06,
    "batch_size"                        : 1,
    "true_batch_size"                   : 64,
    "max_grad_norm"                     : 1.,
    "weight_decay"                      : 1e-5,
    "hidden_dropout"                    : 0.1,
    "attention_dropout"                 : 0.1,

    "include_golden_passage_in_training": False,

    "optimizer"                         : "adam",  # adam, adamw
    "scheduler"                         : None,  # "linear",  # None, linear, cosine, constant
    # "scheduler_warmup_steps": 600,
    # "scheduler_training_steps": 14_400,

    "lookahead_optimizer"               : False,
    # "lookahead_K": 10,
    # "lookahead_alpha": 0.5,

    ###############################
    # Miscellaneous options
    ###############################
    # if training has been discontinued, it can be resumed
    "resume_training"                   : False,
    # "resume_checkpoint": ".saved/generative_reader_EM0.4561_S4500_Mt5-large_2020-11-26_20:38_supergpu7.fit.vutbr.cz",

    # maximum number of training steps
    "max_steps"                         : 10_000,  # on resuming the resumed update steps are counted too
    "save_threshold"                    : 0.41,  # save up some disk space

    # cache where the transformers library will save the models
    "transformers_cache"                : ".Transformers_cache",
    "data_cache_dir"                    : ".data/reader/ranked/",

    "dataset"                           : "nq",

    # number of passages encoded from mini-batch
    #   for training dataset there is always the ground truth passage and the rest is filled with the others
    #   recommended by retriever
    #   for validation dataset only the passages from retriever are used
    "context_length"                    : 25,
    "fp16"                              : False,  # currently does not work for T5
    }
