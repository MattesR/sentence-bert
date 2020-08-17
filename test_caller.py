#!/usr/bin/env python

import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("dataset")
    parser.add_argument("model_name")
    parser.add_argument("pooling")
    parser.add_argument("generator_batch_size")
    parser.add_argument("csv_batch_size")
    parser.add_argument("k")
    args = parser.parse_args()
    os.execv('./index_generator.py', ['filename', 'start', args.generator_batch_size, args.name, args.dataset,
                                               args.model_name, args.pooling])
    print('done')
    os.execv('./experiments.py', ['filename', args.name, args.csv_batch_size, args.k])
