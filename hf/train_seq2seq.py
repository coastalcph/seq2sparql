#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on text translation.
"""
# You can also adapt this script on your own text translation task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random

from transformers.utils.dummy_pt_objects import EncoderDecoderModel
from tokenizers.processors import TemplateProcessing

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import (
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    AdamW,
    BertTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)


logger = logging.getLogger(__name__)

# Parsing input arguments
def parse_args():

    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a translation task"
    )
    parser.add_argument(
        "--predict_with_generate",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A json file containing the training data.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help="Whether to pad all samples to model maximum sentence "
        "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
        "efficient on GPU but very bad for TPU.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A json file containing the validation data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A json file containing the test data.",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to "
        "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=True,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_patience",
        type=int,
        default=math.inf,
        help="Maximum patience before early-stopping.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--data_split", type=str, default=None, help="Data split type.")
    parser.add_argument(
        "--pretrained_encoder", action="store_true", help="Use a pretrained encoder."
    )
    parser.add_argument(
        "--pretrained_encoder_freeze_emb",
        action="store_true",
        help="Freeze pretrained encoder's embedding matrix.",
    )
    parser.add_argument(
        "--only_eval", action="store_true", help="Run only prediction pipeline."
    )
    parser.add_argument(
        "--train_lang", type=str, default="en", help="Language to train on."
    )
    parser.add_argument(
        "--eval_lang", type=str, default="en", help="Language to evaluate on."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    args = parser.parse_args()

    # Sanity checks
    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["json"], "`train_file` should be a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["json"], "`validation_file` should be a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    # Parse the arguments
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    def _init_special_tokens(tokenizer):
        tokenizer.cls_token = "[CLS]"
        tokenizer.bos_token = "[CLS]"
        tokenizer.eos_token = "[SEP]"
        tokenizer.mask_token = "[MASK]"
        tokenizer.pad_token = "[PAD]"
        tokenizer.sep_token = "[SEP]"
        tokenizer.unk_token = "[UNK]"
        return tokenizer

    bert = {
        "tiny": "google/bert_uncased_L-2_H-128_A-2",
        "mini": "google/bert_uncased_L-4_H-256_A-4",
        "small": "google/bert_uncased_L-4_H-512_A-8",
        "medium": "google/bert_uncased_L-8_H-512_A-8",
        "base": "google/bert_uncased_L-12_H-768_A-12",
        "large": "google/bert_uncased_L-24_H-1024_A-16",
    }

    encoder_name = (
        "bert-base-multilingual-cased" if args.pretrained_encoder else bert["base"]
    )
    decoder_name = bert["base"]
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_name, decoder_name
    )
    if not args.pretrained_encoder:
        model.encoder.init_weights()
        src_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"./tokenizers/{args.data_split}/nl-{args.train_lang}-tokenizer.json"
        )
        src_tokenizer = _init_special_tokens(src_tokenizer)
    else:
        if args.pretrained_encoder_freeze_emb:
            for param in model.encoder.embeddings.parameters():
                param.requires_grad = False
        src_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model.decoder.init_weights()
    tgt_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"./tokenizers/{args.data_split}/sparql-{args.train_lang}-tokenizer.json"
    )
    tgt_tokenizer = _init_special_tokens(tgt_tokenizer)
    _pad_token = tgt_tokenizer.pad_token_id
    
    # model.config.decoder_start_token_id = tgt_tokenizer.cls_token_id
    # model.config.pad_token_id = tgt_tokenizer.pad_token_id
    model.encoder.resize_token_embeddings(len(src_tokenizer))
    model.decoder.resize_token_embeddings(len(tgt_tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    padding = "max_length" if args.pad_to_max_length else False

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = [ex["src"] for ex in examples["translation"]]
        targets = [ex["tgt"] for ex in examples["translation"]]
        model_inputs = src_tokenizer(
            inputs, max_length=args.max_source_length, padding=padding, truncation=True
        )
        labels = tgt_tokenizer(
            targets, max_length=max_target_length, padding=padding, truncation=True
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by _pad_token when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tgt_tokenizer.pad_token_id else _pad_token) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = (
        _pad_token if args.ignore_pad_token_for_loss else tgt_tokenizer.pad_token_id
    )
    if args.pad_to_max_length:
        # If padding was already done to max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tgt_tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def _evaluate(dataloader, use_beams=False):
        metric = load_metric("sacrebleu")
        model.eval()
        all_preds = []
        correct = 0
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                generate = (
                    model.generate
                    if getattr(model, "generate", None)
                    else model.module.generate
                )
                generated_tokens = generate(
                    batch["input_ids"],
                    max_length=args.val_max_target_length,
                    num_beams=args.num_beams if use_beams else 1,
                    bos_token_id=tgt_tokenizer.cls_token_id,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tgt_tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(
                        batch["labels"], dim=1, pad_index=tgt_tokenizer.pad_token_id
                    )
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
                if args.ignore_pad_token_for_loss:
                    # Replace _pad_token in the labels as we can't decode them.
                    labels = np.where(
                        labels != _pad_token, labels, tgt_tokenizer.pad_token_id
                    )
                decoded_preds = tgt_tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                decoded_labels = tgt_tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )
                correct += sum(
                    [p == l[0] for p, l in zip(decoded_preds, decoded_labels)]
                )
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                all_preds.extend(decoded_preds)
        eval_metric = metric.compute()
        eval_metric["accuracy"] = correct / len(all_preds)
        return all_preds, eval_metric

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    if not args.only_eval:
        # Train!
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(args.max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0

        best_acc = -1.0
        patience = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(
                    input_ids=batch["input_ids"],
                    decoder_input_ids=batch["labels"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                if completed_steps >= args.max_train_steps:
                    break

            if epoch % 1 == 0:
                decoded_preds, eval_metric = _evaluate(eval_dataloader)
                logger.info(
                    {"bleu": eval_metric["score"], "acc.": eval_metric["accuracy"]}
                )
                if best_acc < eval_metric["accuracy"]:
                    patience = 0
                    best_acc = eval_metric["accuracy"]
                    accelerator.wait_for_everyone()
                    accelerator.unwrap_model(model).save_pretrained(
                        args.output_dir, save_function=accelerator.save
                    )
                else:
                    patience += 1
                    if patience == args.max_patience:
                        break

    model = EncoderDecoderModel.from_pretrained(args.output_dir)
    model = accelerator.prepare(model)
    decoded_preds, eval_metric = _evaluate(test_dataloader)
    logger.info({"bleu": eval_metric["score"], "acc.": eval_metric["accuracy"]})
    with open(
        os.path.join(args.output_dir, f"predictions_{args.eval_lang}.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for line in decoded_preds:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
