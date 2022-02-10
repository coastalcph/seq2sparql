import json
import os
import argparse

from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing


nl_file_path = "tokenizers/tmp/nl.txt"
sparql_file_path = "tokenizers/tmp/sparql.txt"


def write_vocab(vocab, path):
    vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(path, "w", encoding="utf-8") as f:
        for v in vocab:
            f.write(v[0] + "\n")


def make_training_files(paths):
    nl_lines, sparql_lines = [], []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line.strip())["translation"]
                nl_lines.append(line["src"])
                sparql_lines.append(line["tgt"])

    with open(nl_file_path, "w", encoding="utf-8") as f:
        for line in nl_lines:
            f.write(line + "\n")
    with open(sparql_file_path, "w", encoding="utf-8") as f:
        for line in sparql_lines:
            f.write(line + "\n")


def train_bert(file_path):
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=[file_path], min_frequency=1)
    return tokenizer


def train_word_level(file_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.train([file_path], trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train natural language and sparql tokenizers."
    )
    parser.add_argument("--files", nargs="+", required=True, help="Files for training")
    parser.add_argument("--data_split", type=str, required=True, help="Data split type")
    parser.add_argument("--lang", type=str, required=True, help="Language")
    args = parser.parse_args()

    os.makedirs("tokenizers/tmp", exist_ok=True)
    os.makedirs(f"tokenizers/{args.data_split}", exist_ok=True)
    make_training_files(args.files)

    nl_tokenizer = train_word_level(nl_file_path)
    sparql_tokenizer = train_word_level(sparql_file_path)
    nl_tokenizer.save(f"tokenizers/{args.data_split}/nl-{args.lang}-tokenizer.json")
    write_vocab(nl_tokenizer.get_vocab(), f"tokenizers/{args.data_split}/nl-{args.lang}-vocab.txt")
    sparql_tokenizer.save(f"tokenizers/{args.data_split}/sparql-{args.lang}-tokenizer.json")
    write_vocab(
        sparql_tokenizer.get_vocab(), f"tokenizers/{args.data_split}/sparql-{args.lang}-vocab.txt"
    )


if __name__ == "__main__":
    main()