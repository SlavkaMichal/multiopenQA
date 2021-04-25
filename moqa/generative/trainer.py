import copy
import json
import os
import socket
import time

from click import confirm
import torch
import torch.nn.functional as F
import transformers
from torch.nn import DataParallel
from torch.optim import Optimizer, Adam, AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torchtext.data import Iterator
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput

from moqa.datasets.mt5datasets import MT5Dataset
from moqa.common.model_utils import count_parameters, sum_parameters, report_parameters
from moqa.common.utils import timestamp
from moqa.common.eval_utils import metric_max_over_ground_truths, exact_match_score
from moqa.generative.model import MT5QA, T5QA
from moqa.db import PassageDB
from moqa.translate import Translator
from moqa.common import config as logging_cfg
import ipdb
import subprocess
import logging

# from .config_types import TrainConfig

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=logging_cfg.log_file,
    level=logging_cfg.log_level)


def get_model(m):
    if type(m) == DataParallel:
        return m.module
    return m


class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.best_em = config["save_em_threshold"]
        self.increase_validation_frequency = False
        # adding special tokens
        self.tokenizer = self.init_tokenizer(config['reader_tokenizer_type'], config['cache_transformers'])

        self.db = PassageDB(db_path=self.config['database'])

        self.translator = Translator(device='cpu')  # I don't thing that this will fit in GPU memory

        # log git commit
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True)
        commit = out.stdout.decode('utf-8')[:-1]
        if out.returncode == 0:
            logging.info(f"Commit hash: {commit}")

    @staticmethod
    def init_tokenizer(tokenizer_type, cache_dir) -> PreTrainedTokenizer:
        """
        Creates tokenizer and add special tokens into it
        """
        reader_tokenizer = AutoTokenizer.from_pretrained(tokenizer_type,
                                                         cache_dir=cache_dir)
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

        train, val, test = self.load_data()
        # inspect data
        if config['interactive']:
            ipdb.set_trace()

        if config['pretrained_model'] is None:
            logging.info("Loading model")
        else:
            logging.info(f"Loading model from {config['pretrained_model']}")

        if config['pretrained_model'] is not None:
            model = torch.load(config["pretrained_model"], map_location=self.device)
        elif 'mt5' in config['reader_transformer_type']:
            model = MT5QA.from_pretrained(config).to(self.device)
        else:
            model = T5QA.from_pretrained(config).to(self.device)

        logging.info(f"Training data examples:{len(train)}")
        if val is not None:
            logging.info(f"Validation data examples:{len(val)}")
        if test is not None:
            logging.info(f"Test data examples {len(test)}")

        if config['interactive'] and not confirm("Data are prepared. Do you wish to start training?", default=True):
            raise KeyboardInterrupt

        if not config["test_only"]:
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

        if test is not None:
            test_iter = Iterator(test,
                                 sort=False, shuffle=False,
                                 batch_size=1,
                                 repeat=False, device=self.device)
        else:
            test_iter = None

        logging.info(f"Resizing token embeddings to length {len(self.tokenizer)}")
        # model.resize_token_embeddings(len(self.tokenizer))

        logging.info(f"Model has {count_parameters(model)} trainable parameters")
        logging.info(f"Trainable parameter checksum: {sum_parameters(model)}")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        # logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")
        logging.debug(f"Model input embeddings:{model.get_input_embeddings()}")
        logging.debug(f"Model output embeddings:{model.get_output_embeddings()}")

        # Init optimizer
        if not config["test_only"]:
            if test is not None:
                logging.info("Validating before start")
                self.validate(model, test_iter)
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
            if self.config["optimizer"] == "adamw":
                optimizer = AdamW
            elif self.config["optimizer"] == "adam":
                optimizer = Adam
            else:
                raise ValueError("Unsupported optimizer")
            optimizer = optimizer(optimizer_grouped_parameters,
                                  lr=self.config["learning_rate"],
                                  eps=self.config["adam_eps"])

            if config['pretrained_model'] is not None and config['load_optimizer_state_dict']:
                optimizer.load_state_dict(model.optimizer_state_dict)
                del model.optimizer_state_dict
            if hasattr(model, 'optimizer_state_dict'):
                # if optimizer is not being loaded still delete optimizer state dict
                del model.optimizer_state_dict

            # Init scheduler
            if self.config["scheduler_warmup_steps"] > 0:
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
            logging.info(f"Steps: {get_model(model).training_steps}")
            try:
                it = 0
                while get_model(model).training_steps < self.config["max_steps"]:
                    logging.info(f"Epoch {it}")
                    train_loss = self.train_epoch(model=model,
                                                  epoch=it,
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
                if config['interactive'] and confirm("Save model parameters?", default=False):
                    serializable_model_name = self.config['reader_transformer_type'].replace("/", "_")
                    saveable_model = get_model(model)
                    saved_name = os.path.join(self.config['save_dir'], f"generative_reader_"
                                                                       f"last_"
                                                                       f"S{get_model(model).training_steps}_"
                                                                       f"M{serializable_model_name}_"
                                                                       f"{timestamp()}_{socket.gethostname()}")
                    self.best_ckpt_name = saved_name
                    torch.save(saveable_model, saved_name)

        logging.info("#" * 50)
        if test is not None:
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
                    epoch: int,
                    data_iter: Iterator,
                    val_iter: Iterator,
                    optimizer: Optimizer,
                    scheduler: LambdaLR):
        #  Training flags
        model.train()
        # Make sure parameters are zero
        optimizer.zero_grad()
        if self.config["fp16"]:
            grad_scaler = GradScaler()

        # Determine update ratio, e.g. if true_batch_size = 32 and batch_size=8, then
        # gradients should be updated  every 4th iteration (except for last update!)
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        assert self.config["true_batch_size"] % self.config["batch_size"] == 0
        updated = False

        # Calculate total number of updates per epoch
        total = (len(data_iter.data()) // data_iter.batch_size + 1) - 1
        logging.info(f"Total number of examples: {total}")

        it = tqdm(enumerate(data_iter), total=total)

        # For progressive  training loss  reporting
        total_losses = []
        losses_per_update = []

        for i, batch in it:
            assert len(batch.src) == 1  # more  than 1 example per batch is unsupported
            updated = False
            if len(losses_per_update) == 0 and total - i < update_ratio:
                update_ratio = total - i
                logging.info(f"Adjusting for last update, ratio: {update_ratio}, i: {i}")

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

                if self.config['fp16']:
                    with autocast():
                        outputs = model(input_ids=batch.src, attention_mask=batch.src_mask,
                                        passage_mask=batch.doc_mask,
                                        decoder_input_ids=batch.target[:, :-1].contiguous(),
                                        decoder_attention_mask=batch.target_mask[:, :-1].contiguous(),
                                        use_cache=False)
                        lm_logits = outputs[0]
                        labels = batch.target[:, 1:].reshape(-1)

                        loss = F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                               reduction='mean')
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

                    loss = F.cross_entropy(lm_logits.view(-1, get_model(model).config.vocab_size), labels,
                                           reduction='mean')
                    loss /= update_ratio
                    loss.backward()

                # record losses to list
                losses_per_update.append(loss.item())

                if len(losses_per_update) == update_ratio:
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
                        scheduler.step(epoch=epoch)
                    updated = True
                    # If we are past 2/3 of expected training steps
                    if get_model(model).training_steps > (2 * self.config["max_steps"] / 3) and \
                            not self.increase_validation_frequency and self.config["validate_after_steps"] > 20:
                        # Increase validation frequency with factor of 2
                        self.config["validate_after_steps"] = self.config["validate_after_steps"] // 2
                        self.increase_validation_frequency = True
                        logging.info(f"Validation frequency increased to: {self.config['validate_after_steps']}")

                    it.set_description(f"Steps: {get_model(model).training_steps}, Training loss: {loss_per_update}")

                    # Validate after every validate_after_steps steps
                    if get_model(model).training_steps > 1 and \
                            get_model(model).training_steps % self.config["validate_after_steps"] == 0:
                        logging.info(f"Training loss: {loss_per_update}")
                        logging.info(f"AVG loss: {sum(total_losses) / len(total_losses)}")
                        self.validate(model, val_iter, optimizer_dict=optimizer.state_dict())
                else:
                    updated = False
            # Catch out-of-memory errors
            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    print(torch.cuda.memory_stats())
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
            # Do the last step if needed 48 49
            logging.info(f"Last update {len(losses_per_update), update_ratio}")
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                           self.config["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
            get_model(model).training_steps += 1
            if scheduler is not None:
                scheduler.step(epoch=epoch)

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

        translated_hits = {}
        translated_total = {}
        if 'mt5' not in self.config['reader_transformer_type']:
            for lang in self.config['languages']:
                translated_hits[lang] = 0
                translated_total[lang] = 0

        loss_list = []
        if self.config['log_results']:
            import csv
            model_type = self.config['reader_transformer_type'].replace("/", "_")
            steps = get_model(model).training_steps
            outf = open(f"{self.config['results']}/gen_reader_{model_type}_{steps}.csv", "w", encoding="utf-8")
            csvw = csv.writer(outf, delimiter=',')
            csvw.writerow(["Correct", "Question", "Predicted Answer", "GT Answer", "Input"])
        for j, batch in it:
            batch.src = batch.src[0]
            batch.src_mask = batch.src_mask[0]
            batch.doc_mask = batch.doc_mask[0] if hasattr(batch, "doc_mask") else None
            batch.lang = batch.lang[0]

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

            if 'mt5' not in self.config['reader_transformer_type'] and batch.lang != 'en':
                predicted_answers_translated = self.translator.from_en(predicted_answers, [batch.lang])

            for i in range(len(batch)):
                hit = metric_max_over_ground_truths(
                    metric_fn=exact_match_score, prediction=predicted_answers[i],
                    ground_truths=batch.answers[i])
                hits += int(hit)
                if 'mt5' not in self.config['reader_transformer_type'] and batch.lang != 'en':
                    translated_hit = metric_max_over_ground_truths(
                        metric_fn=exact_match_score, prediction=predicted_answers_translated[i],
                        ground_truths=batch.answers[i])
                    translated_hits[batch.lang[i]] += int(translated_hit)
                    translated_total[batch.lang[i]] += 1

                if self.config['log_results']:
                    csvw.writerow([
                        hit,
                        batch.question[i],
                        predicted_answers[i],
                        batch.answers[i],
                        self.tokenizer.decode(batch.src[i])
                        ])

            it.set_description(f"Val Loss: {sum(loss_list) / len(loss_list):.3f} EM: {hits / total:.3f}")

        EM = hits / total
        logging.info(f"S: {get_model(model).training_steps} Validation Loss: {sum(loss_list) / len(loss_list)}")
        logging.info(f"Validation EM: {EM}")
        if 'mt5' not in self.config['reader_transformer_type'] and batch.lang != 'en':
            for lang, lang_hits in translated_hits.items():
                lang_total = translated_total[lang]
                EM = lang_hits / lang_total
                logging.info(f"Validation EM {lang}: {EM} ({lang_total}/{lang_hits})")

        if self.config['log_results']:
            outf.close()
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
        model.train()
        return EM

    def load_data(self):
        config = self.config

        include_passage_masks = config["fusion_strategy"] == "passages"

        model_name = config['reader_transformer_type'].replace('/', '-')
        # model_name = model_name if model_name == 't5-small' else ""

        train = None
        val = None
        test = None

        if config['cached_data'] is None:
            logging.info("Loading from cache")
            if config['data_size'] <= 0 and config['test_only']:
                # if test only and limit for data size was not set reduce the amount of data
                config["data_size"] = 10_000

            if not confirm("WARNING: Not sure if this work, you probably should look at it.", default=False):
                raise KeyboardInterrupt
            data = MT5Dataset(
                datafile=config["data"],
                preprocess=config["preprocess"],
                model_name=model_name,
                tokenizer=self.tokenizer,
                db_multi=self.db,  # database with passages
                langs=config["languages"],
                max_context_size=config["max_context_size"],
                interactive=config["interactive"],
                multi_lingual_query=config["multi_lingual_query"],  # use multiple languages per question
                multi_lingual_answer_lang_code=config["multi_lingual_answer_lang_code"],
                translated_query=config["translated_query"],  # use translated questions
                translated_retrieval_search=config['translated_retrieval_search'],
                english_ctxs_only=config["english_ctxs_only"],
                include_golden_passage=config["include_golden_passage"],
                use_dpr_golden=config["use_dpr_golden"],
                # if available use dpr for golden passage if include_golden_passage is true
                only_gt_passages=config["only_gt_passages"],
                examples_per_sample=config["examples_per_sample"],
                # creates multiple version of a sample but in different languages
                max_len=config["max_len"],
                data_size=config["data_size"],  # limit number of examples for debugging
                is_training=True,  # does not tokenize answers
                preprocessing_truncation="truncate_only_passages",  # truncation strategy
                include_passage_masks=include_passage_masks,
                use_cache=True,  # use cached examples
                cached_data_path=None,  #
                cache_dir=config["cache_data"],
                device=self.device
                )

            logging.info(f"Total data examples:{len(data)}")
            # return splits (train, test, val?), irrespectively of split_ratio
            splits = data.split(split_ratio=config['split_ratio'])
            # splits = data.split(split_ratio=config['split_ratio'], stratified=True, strata_field='lang')
            if len(splits) == 3:
                train = splits[0]
                val = splits[2]
                test = splits[1]
            elif len(splits) == 2:
                train = splits[0]
                val = splits[1]
                test = None
            elif config["test_only"]:
                # should be an error
                val = None
                train = None
                test = data
            else:
                raise RuntimeError("Something went wrong with data preparation")
        else:
            logging.info("Loading from splits")
            if not config['test_only']:
                train = MT5Dataset(config["data"]['train'],
                                   preprocess=config["preprocess"],
                                   model_name=model_name,
                                   tokenizer=self.tokenizer,
                                   db_multi=self.db,  # database with passages
                                   langs=config["languages"],
                                   max_context_size=config["max_context_size"],
                                   interactive=config["interactive"],
                                   multi_lingual_query=config["multi_lingual_query"],
                                   multi_lingual_answer_lang_code=config["multi_lingual_answer_lang_code"],
                                   # use multiple languages per question
                                   translated_query=config["translated_query"],  # use translated questions
                                   translated_retrieval_search=config['translated_retrieval_search'],
                                   english_ctxs_only=config["english_ctxs_only"],
                                   include_golden_passage=config["include_golden_passage"],
                                   use_dpr_golden=config["use_dpr_golden"],
                                   # if available use dpr for golden passage if include_golden_passage is true
                                   only_gt_passages=config["only_gt_passages"],
                                   examples_per_sample=config["examples_per_sample"],
                                   # creates multiple version of a sample but in different languages
                                   max_len=config["max_len"],
                                   data_size=config["data_size"],  # limit number of examples for debugging
                                   is_training=True,  # does not tokenize answers
                                   preprocessing_truncation="truncate_only_passages",  # truncation strategy
                                   include_passage_masks=include_passage_masks,
                                   use_cache=True,  # use cached examples
                                   cached_data_path=None if 'train' not in config['cached_data'] else
                                   config['cached_data']['train'],  #
                                   cache_dir=config["cache_data"],
                                   device=self.device
                                   )
                val = MT5Dataset(config["data"]['val'],
                                 preprocess=config["preprocess"],
                                 model_name=model_name,
                                 tokenizer=self.tokenizer,
                                 db_multi=self.db,  # database with passages
                                 langs=config["languages"],
                                 max_context_size=config["max_context_size"],
                                 interactive=config["interactive"],
                                 multi_lingual_query=config["multi_lingual_query"],
                                 multi_lingual_answer_lang_code=config["multi_lingual_answer_lang_code"],
                                 # use multiple languages per question
                                 translated_query=config["translated_query"],  # use translated questions
                                 translated_retrieval_search=config['translated_retrieval_search'],
                                 english_ctxs_only=config["english_ctxs_only"],
                                 include_golden_passage=config["include_golden_passage"],
                                 use_dpr_golden=config["use_dpr_golden"],
                                 # if available use dpr for golden passage if include_golden_passage is true
                                 only_gt_passages=config["only_gt_passages"],
                                 examples_per_sample=config["examples_per_sample"],
                                 # creates multiple version of a sample but in different languages
                                 max_len=config["max_len"],
                                 data_size=config["data_size"],  # limit number of examples for debugging
                                 is_training=True,  # does not tokenize answers
                                 preprocessing_truncation="truncate_only_passages",  # truncation strategy
                                 include_passage_masks=include_passage_masks,
                                 use_cache=True,  # use cached examples
                                 cached_data_path=None if 'val' not in config['cached_data'] else
                                 config['cached_data']['val'],  #
                                 cache_dir=config["cache_data"],
                                 device=self.device
                                 )
            if 'test' in config['data']:
                test = MT5Dataset(config["data"]['test'],
                                  irrelevant_passage_langs=config['test_irrelevant_passage_langs'],
                                  preprocess=config["preprocess"],
                                  model_name=model_name,
                                  tokenizer=self.tokenizer,
                                  db_multi=self.db,  # database with passages
                                  langs=config["languages"],
                                  max_context_size=config["max_context_size"],
                                  interactive=config["interactive"],
                                  multi_lingual_query=config["multi_lingual_query"],
                                  multi_lingual_answer_lang_code=config["multi_lingual_answer_lang_code"],
                                  english_ctxs_only=config["english_ctxs_only"],
                                  translated_query=config["test_translated_query"],  # use translated questions
                                  translated_retrieval_search=config["test_translated_retrieval_search"],
                                  include_golden_passage=config["test_include_golden_passage"],
                                  use_dpr_golden=config["test_use_dpr_golden"],
                                  only_gt_passages=config["test_only_gt_passages"],
                                  examples_per_sample=len(config['languages']),
                                  max_len=config["max_len"],
                                  data_size=config["data_size"],  # limit number of examples for debugging
                                  is_training=True,  # does not tokenize answers
                                  preprocessing_truncation="truncate_only_passages",  # truncation strategy
                                  include_passage_masks=include_passage_masks,
                                  use_cache=True,  # use cached examples
                                  cached_data_path=None if 'test' not in config['cached_data'] else
                                  config['cached_data']['test'],  #
                                  cache_dir=config["cache_data"],
                                  device=self.device)

        return train, val, test
