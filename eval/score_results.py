import hashlib
import utils
import argparse
import os
from random import shuffle
from collections import defaultdict
import json

SCORED_HASHES = 'hashes.txt'
SCORES = 'scores.txt'

MIN_SCORE = '0'
MAX_SCORE = '5'

def main(args):
  results_files = os.listdir(args.results_dir)

  # Load hashes of results that have been labeled already.
  try:
    with open(SCORED_HASHES, 'r') as f:
      labeled_hashes = json.load(f)
      print(f'Found {len(labeled_hashes)} previously labeled examples.')
  except:
    labeled_hashes = []

  # Read in results files.
  print("Reading in results files:")
  unlabeled_outputs = {}
  for results_file in results_files:
    outputs = utils.parse_results_file(
      os.path.join(args.results_dir, results_file))
    print(results_file.encode('utf-8'))
    file_hash = hashlib.md5(results_file.encode('utf-8')).hexdigest()
    for src_hash in outputs:
      if src_hash not in labeled_hashes:
        output = outputs[src_hash]
        output['file_hash'] = file_hash
        unlabeled_outputs[src_hash] = output

  scores = defaultdict(lambda: defaultdict(int))
  # Load hashes of results that have been labeled already.
  try:
    with open(SCORES, 'r') as f:
      scores_dict = json.load(f)
      num_scores = sum([sum(scores_dict[file].values()) for file in scores_dict])
      print(f'Found {num_scores} scores in {len(scores_dict)} files.')
      for file in scores_dict:
        for label in scores_dict[file]:
          scores[file][label] = scores_dict[file][label]
  except:
    pass

  num_scores = sum([sum(scores[file].values()) for file in scores])
  assert num_scores == len(labeled_hashes)

  if args.show_results:
    print()
    response = input('Are you sure you want to view the results? (y/n) ')
    if response.lower() == 'y':
      os.system('clear')
      for results_file in results_files:
        file_hash = hashlib.md5(results_file.encode('utf-8')).hexdigest()
        file_scores = scores[file_hash]
        print(results_file)
        print(dict(file_scores))
    exit()

  # Suffle the source hashes (randomizes the order that sentences are shown).
  hashes = list(unlabeled_outputs.keys())
  shuffle(hashes)
  i = 0

  print()
  while True:
    unlabeled_output = unlabeled_outputs[hashes[i]]
    print(f'Source:\t\t{unlabeled_output["src"]}')
    print(f'Prediction:\t{unlabeled_output["pred"]}')

    # Get a score from the user.
    score = input(
      """Rate the quality of the output on a scale of %s-%s.
         0: No change
         1: TODO
         2: TODO
         3: TODO
         4: TODO
         5: TODO
         Enter -1 to exit.\n""" % (MIN_SCORE, MAX_SCORE))
    while (score < MIN_SCORE or MAX_SCORE < score) and score != '-1':
      score = input(f'Invalid score. Rate the quality of the output on a '
                    f'scale of {MIN_SCORE}-{MAX_SCORE}. Enter -1 to exit. ')
    if score == '-1':
      break

    # Update the data structures that keep track of scores.
    scores[unlabeled_output['file_hash']][score] += 1
    labeled_hashes.append(hashes[i])
    i += 1
    os.system('clear')

  # Save data structures.
  with open(SCORES, 'w') as f:
    print("Saving scores.")
    json.dump(scores, f)

  with open(SCORED_HASHES, 'w') as f:
    print("Saving scored examples.")
    json.dump(labeled_hashes, f)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--results_dir",
    help="Directory that contains the inference results.",
    required=True
  )

  parser.add_argument(
    "--show_results",
    help="If true, display the inference results.",
    action="store_true"
  )

  main(parser.parse_args())