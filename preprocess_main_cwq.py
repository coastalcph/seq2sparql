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

"""Preprocesses a specific split of the CFQ dataset."""

from absl import app
from absl import flags

import preprocess_cwq as preprocessor

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', None,
                    'path of the dataset.')

flags.DEFINE_string('split_path', None, 'path to the JSON file containing '
                    'split information.')

flags.DEFINE_string('save_path', None, 'Path to the directory where to '
                    'save the files to.')

flags.mark_flag_as_required('save_path')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset = preprocessor.load_dataset(FLAGS.dataset_path, FLAGS.split_path)
  preprocessor.write_dataset(dataset, FLAGS.save_path)
  token_vocab = preprocessor.get_token_vocab(FLAGS.save_path)
  preprocessor.write_token_vocab(token_vocab, FLAGS.save_path)


if __name__ == '__main__':
  app.run(main)
