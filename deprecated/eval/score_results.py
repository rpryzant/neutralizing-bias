"""Command line interface to score inference results.

Example usage:
  python score_results.py --results_dir path/to/inference_outputs --scores_file my_scores_file
"""
import hashlib
import argparse
import os
from random import shuffle, choice
from collections import defaultdict
import json
from tqdm import tqdm

import utils



MIN_SCORE = '1'
MAX_SCORE = '2'

def detokenize(s):
    out = []
    for w in s.split():
        if w.startswith('##') and len(out) > 0:
            out[-1] += w[2:]
        else:
            out.append(w)
    return ' '.join(out)


def main(args):

  # Load save file
  try:
    with open(args.scores_file, 'r') as f:
      # Load hashes of results that have been labeled already.
      tmp = json.load(f)
      scores_dict = defaultdict(lambda: defaultdict(int))
      for filename in tmp:
        for src_hash in tmp[filename]:
          scores_dict[filename][src_hash] = tmp[filename][src_hash]

      labeled_hashes = set([
        src_hash for file_dict in scores_dict.values() for src_hash in file_dict.keys()
      ])
      print(f'Found {len(labeled_hashes)} previously labeled examples.')
  except:
    labeled_hashes = set()
    scores_dict = defaultdict(lambda: defaultdict(int))

  # Load results files
  results_files = os.listdir(args.results_dir)
  results_dict = {
    results_path: utils.parse_results_file(
      os.path.join(args.results_dir, results_path), ignore_unchanged=True)
    for results_path in results_files
  }

  # Get examples that haven't been labeled yet
  num_unlabeled_outputs = len([
    (filename, src_hash) 
    for filename in results_dict.keys()
    for src_hash in results_dict[filename].keys()
    if src_hash not in labeled_hashes
  ])
  print(f'Found {num_unlabeled_outputs} examples to label.')

  # label examples (random uniform across files)
  i = 0
  while True:
    filename = choice(list(results_dict.keys()))
    for _ in range(100): #hacky: go until you get a cache miss
      src_hash = choice(list(results_dict[filename].keys()))
      if src_hash not in labeled_hashes:
        break
    if src_hash in labeled_hashes:
      continue

    unlabeled_output = results_dict[filename][src_hash]
    print('%d / %d' % (i, num_unlabeled_outputs))
    print(f'Source:\t\t{detokenize(str(unlabeled_output["src"]))}')
    print(f'Prediction:\t{detokenize(str(unlabeled_output["pred"]))}')

    # Get a score from the user.
    score = input(
      """Rate the quality of the output on a scale of %s-%s.
         1: Unsuccessful
         2: Successful
         Enter -1 to exit.\n""" % (MIN_SCORE, MAX_SCORE))
    while (score < MIN_SCORE or MAX_SCORE < score) and score != '-1':
      score = input(f'Invalid score. Rate the quality of the output on a '
                    f'scale of {MIN_SCORE}-{MAX_SCORE}. Enter -1 to exit. ')
    if score == '-1':
      break

    # Update the data structures that keep track of scores.
    labeled_hashes.add(src_hash)
    scores_dict[filename][src_hash] = score
    os.system('clear')
    i += 1

  # Save data structures.
  with open(args.scores_file, 'w') as f:
    print("Saving scores.")
    json.dump(scores_dict, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--results_dir",
    help="Directory that contains the inference results.",
    required=True
  )

  parser.add_argument(
    "--scores_file",
    help="File to dump scores into.",
    default='scores.txt',
    required=True
  )

  main(parser.parse_args())
