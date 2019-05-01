"""Command line interface to score inference results.

Example usage:
  python generate_media_turk_csv.py --results_dir path/to/inference_outputs
"""
import argparse
import csv
import os

from random import shuffle

import utils


def main(args):
  # Load results files
  results_files = os.listdir(args.results_dir)
  sentences = []
  for results_path in results_files:
    with open(os.path.join(args.results_dir, results_path)) as file:
      for line in file:
        sentences.append((line.split('\t')[3], results_path))
  
  shuffle(sentences)
  print(sentences[:5])
  print(sentences[-5:])

  with open('media_turk_data.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['ID', 'Sentence', 'Source'])
    for i, (sentence, filename) in enumerate(sentences):
      writer.writerow([i, sentence, filename])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--results_dir",
    help="Directory that contains the inference results.",
    required=True
  )

  main(parser.parse_args())
