# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utils for preprocessing the CFQ dataset."""

import collections
import json
import os
import string
import jieba
from typing import Dict, List, Tuple

from absl import logging

from tensorflow.compat.v1.io import gfile

Dataset = Dict[str, List[Tuple[str, str]]]


def tokenize(text, data_language):
    if data_language != 'zh':
        text = map(lambda c: ' %s ' % c if c in string.punctuation else c, text)
        return ' '.join(''.join(text).split())
    else:
        text = jieba.cut(text, cut_all=False)
        return " ".join(text)


def preprocess_sparql(query):
    """Do various preprocessing on the SPARQL query."""
    # Tokenize braces.
    query = query.replace('count(*)', 'count ( * )')

    tokens = []
    for token in query.split():
        # Replace 'ns:' prefixes.
        if token.startswith('ns:'):
            token = token[3:]
        # Replace mid prefixes.
        if token.startswith('m.'):
            token = 'm_' + token[2:]
        tokens.append(token)

    return ' '.join(tokens).replace('\\n', ' ')


def load_dataset(dataset, split, data_language):
    """Load dataset do some basic preprocessing.
    Change variable passed to tokenize_punctuation for other languages"""
    logging.info('Loading dataset.')
    with open(dataset, 'r') as f_1:
        cwq = json.load(f_1)
    with open(split, 'r') as f_2:
        splits = json.load(f_2)
    split_names = {'train': 'trainIdxs', 'dev': 'devIdxs', 'test': 'testIdxs'}
    questionpatternmodentities_dict = {'en': 'questionPatternModEntities', 'he': 'questionPatternModEntities_he',
                                       'kn': 'questionPatternModEntities_kn', 'zh': 'questionPatternModEntities_zh'}
    dataset = collections.defaultdict(list)
    for cwq_split_name, split_name in split_names.items():
        for idx in splits[split_name]:
            encode_decode_pair = (tokenize(cwq[idx][questionpatternmodentities_dict[data_language]], data_language),
                                  preprocess_sparql(cwq[idx]['sparqlPatternModEntities']))
            dataset[cwq_split_name].append(encode_decode_pair)

    size_str = ', '.join(f'{s}={len(dataset[s])}' for s in split_names)
    logging.info('Finished loading splits. Size: %s', size_str)
    return dataset


def write_dataset(dataset, save_path):
    """Saves the given dataset into the given location."""
    if not dataset:
        logging.info('No dataset to write.')
        return
    logging.info('Writing dataset to %s', save_path)
    for split_name, list_of_input_output_pairs in dataset.items():
        folder_name = os.path.join(save_path, split_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        encode_name = os.path.join(folder_name, '%s_encode.txt' % split_name)
        decode_name = os.path.join(folder_name, '%s_decode.txt' % split_name)
        with gfile.GFile(encode_name,
                         'w') as encode_f, gfile.GFile(decode_name,
                                                       'w') as decode_f:
            for pair in list_of_input_output_pairs:
                encode_f.write(pair[0] + '\n')
                decode_f.write(pair[1] + '\n')
    logging.info('Dataset written to %s', save_path)


def write_token_vocab(words,
                      save_path,
                      problem='cwq'):
    """"Writes token vocabulary from @words to @save_path."""
    # Sort tokens by frequency and then lexically to break ties.
    words_with_counts = words.most_common()
    words_with_counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
    vocab_path = os.path.join(save_path, 'vocab.%s.tokens' % problem)

    with gfile.GFile(vocab_path, 'w') as f:
        # Tensor2tensor needs these additional tokens.
        f.write('<pad>\n<EOS>\n<OOV>\n')
        for word, _ in words_with_counts:
            f.write(word + '\n')
    logging.info('Token vocabulary written to %s (%s distinct tokens).',
                 vocab_path, len(words))


def get_lines(path, filename):
    with gfile.GFile(os.path.join(path, 'train', filename)) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def get_token_vocab(path):
    words = collections.Counter()
    lines = get_lines(path, 'train_encode.txt')
    lines.extend(get_lines(path, 'train_decode.txt'))
    for line in lines:
        words.update(line.split(' '))
    return words
