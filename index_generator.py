#!/usr/bin/env python

import os
import sys
import time
from datetime import datetime
import faiss
import numpy as np
import shutil
from util.embeddings import array_from_list
from experiments import get_dataset
from experiments import generate_embeddings

#  Defaults:
BATCH_SIZE = 8
FAISS_PATH = './faiss_indexes'
INDEX_SIZE = 1000000
DATASET_PATH = './datasets/generated'
valid_model_names = ['bert-base-uncased', 'bert-base-cased', 'bert-base-multilingual-cased',
                     'roberta-base-openai-detector', 'bert-base-german-dbmdz-uncased', 'bert-base-german-dbmdz-cased',
                     'bert-base-multilingual-uncased']
valid_SentenceBert_model_names = ['bert-base-nli-stsb-mean-tokens', 'roberta-base-nli-stsb-mean-tokens',
                                  'distiluse-base-multilingual-cased']
valid_averaging_modes = ['CLS', 'mean', 'SentenceBert']


class IndexGenerator:

    def __init__(self, batch_size, name, data_location, read_from_file, path=FAISS_PATH, model='document_embeddings',
                 pooling='mean', index_size=INDEX_SIZE, index_start=0, index_number=0,
                 fail_mode=0, fail_size=0, previous_time=0., failed_list=None):
        # make all arguments class fields
        if not failed_list:
            self.failed_list = []
        else:
            with open(failed_list, 'r') as f:
                self.failed_list = [tuple(map(int, line.split(' '))) for line in f]
        self.model = model
        self.pooling = pooling
        self.fail_size = fail_size
        self.fail_mode = fail_mode
        self.index_number = index_number
        self.index_start = index_start
        self.index_position = index_start  # index start is still needed, that's why it's not increased
        self.index_size = index_size
        self.path = path
        self.read_from_file = read_from_file
        if batch_size:
            self.batch_size = batch_size
        self.name = name
        self.data_location = data_location
        self.previous_time = previous_time
        if not self.path.endswith(f'/{name}'):  #
            self.path += f'/{name}'
        # if index_number is 0, it was started from the user, not from the script itself.
        if self.index_number == 0 and not self.fail_mode:
            try:
                os.mkdir(self.path)
            except FileExistsError:
                print(f'directory already exists and I am just deleting it.')
                shutil.rmtree(self.path)
                os.mkdir(self.path)
            # write basic information into index information File
            self.write_index_information(f'--- Index Information for Testcase {self.name} ---')
            self.write_index_information(f'Dataset: {self.data_location}')
            self.write_index_information(f'Model: {self.model}')
            self.write_index_information(f'Batch Size : {self.batch_size}')
        #  read in the dataset. The dataset is always read completely and spliced afterwards
        self.document_pairs = get_dataset(data_location, read_from_file, write=f'{name}_dataset')

        # creation of the index and the ID-Map which adds the index
        # If I change the behavior to include a fail-mode and pick up, from it, read in the current index
        if self.fail_mode:
            self.id_index = faiss.read_index(f'{path}/{name}_{index_number}')
        else:
            if not self.model == 'distiluse-base-multilingual-cased':
                self.index = faiss.IndexFlatIP(768)  # Metric InnerProduct
            else:
                self.index = faiss.IndexFlatIP(512)  # Metric InnerProduct
            self.id_index = faiss.IndexIDMap(self.index)
        if not failed_list:
            self.failed_list = []
        else:
            with open(failed_list, 'r') as f:
                self.failed_list = [tuple(map(int, line.split(' '))) for line in f]

    def generate_index(self):
        if self.index_number == 0 and not self.fail_mode:
            self.write_index_information(f'Index Generation started on {datetime.now()}')
        start_time = time.time()
        #  the index position is set to index start and is increased
        #  by batch_size (or at the end by however many embeddings were left in the embedding_generator).
        #  It represents the index of the next embeddings which are added.
        # here we start iterating over all batches in the embedding_generator.
        if self.fail_mode == 1 and self.batch_size < 1:
            # print('I am in Fail Mode! <3')
            # create a embedding_generator witch batch size 1 for as many vectors as needed to be analyzed individually.
            fail_generator = generate_embeddings(self.document_pairs[self.index_position:
                                                                     self.index_position + self.fail_size],
                                                 1, offset=self.index_position, model_name=self.model, pooling=self.pooling)
            # only append the index by pairs.
            pair = []
            for batch in fail_generator:
                if batch[0]:  # creating embeddings was successful
                    pair.append(batch[1][0])  # append the array inside the list
                    if len(pair) == 2:
                        # if there's a pair (so two vectors), start the routine to append the pair.
                        self.add_to_index(pair)
                        pair = []
                        self.fail_size -= len(pair)
                else:
                    print(f'index {self.index_position} is not embeddable in single embed mode')
                    failed_entry = (batch[1][0] - (batch[1][0] % 2), batch[1][0] + 1 - (batch[1][0] % 2))
                    self.fail_size -= 2  # here, it will always be a pair, calculating a two is not necessary.
                    self.failed_list.append(failed_entry)
                    if self.fail_size == 0:
                        self.fail_mode = 2
                    self.write_index_information(f'Embeddings {failed_entry[0]}  and {failed_entry[1]} '
                                                 f'are not in the index {self.index_number} from '
                                                 f'{self.id_index.id_map.at(0)} to {self.index_position - 1}\n')
                    faiss.write_index(self.id_index, self.path + f'/{self.name}_{self.index_number}')
                    self.index_start = failed_entry[1] + 1
                    print(f'index start is at {self.index_start}')
                    overall_time = time.time() - start_time + self.previous_time
                    self.restart(overall_time)
                    exit(-1)
            print('leaving fail Mode')
        embedding_generator = generate_embeddings(self.document_pairs[self.index_position:],
                                                  self.batch_size, offset=self.index_position, model_name=self.model,
                                                  pooling=self.pooling)
        for batch in embedding_generator:
            if batch[0]:
                self.add_to_index(batch[1])
            else:
                print(f'failed at index {self.index_position} to {self.index_position + self.batch_size - 1}\n'
                      f'retrying that batch in fail_mode')
                self.failed_list.append(batch[1])
                faiss.write_index(self.id_index, self.path + f'/{self.name}_{self.index_number}')
                self.index_start = self.index_position # same as before the failed list, don't step over the batch now!
                self.fail_mode = 1
                overall_time = time.time() - start_time + self.previous_time
                self.restart(overall_time)
                exit(-1)
        overall_time = time.time() - start_time + self.previous_time
        faiss.write_index(self.id_index, self.path + f'/{self.name}_{self.index_number}')
        self.write_index_information(f'Index {self.name}_{self.index_number} contains embeddings '
                                     f'from {self.id_index.id_map.at(0)} to {self.index_position - 1}\n'
                                     f'time for finalizing the index (without module loading): {overall_time}')
        stop_date_time = datetime.now()
        self.write_index_information(f'Index Creation stopped at {stop_date_time}')
        with open(self.path + '/Index Information.txt', 'r') as f:
            for line in f:
                if line.startswith('Index Generation started on'):
                    start_date_string = line[len('Index Generation started on '):]
                    start_date_string = start_date_string.rstrip()
                    starte_date_time = datetime.strptime(start_date_string, '%Y-%m-%d %H:%M:%S.%f')
        self.write_index_information(f'time for finalizing the Index (real total):'
                                     f' {(stop_date_time-starte_date_time).total_seconds()}')

    def add_to_index(self, batch):
        array_vectors = array_from_list(batch)
        # normalization before addition to the index, as mentioned here:
        # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distanc
        faiss.normalize_L2(array_vectors)
        self.id_index.add_with_ids(array_vectors,
                                   np.asarray([j for j in range(self.index_position,
                                                                self.index_position + len(batch))])
                                   )
        self.index_position += len(batch)
        if self.index_position - self.index_start >= self.index_size:
            #  creation of index_information file. If it exists, append, if not, write

            self.write_index_information(f'Index {self.name}_{self.index_number} contains embeddings '
                                         f'from {self.index_start} to {self.index_position - 1}\n')
            faiss.write_index(self.id_index, self.path + f'/{self.name}_{self.index_number}')
            self.index_number += 1
            self.index_start = self.index_position
            self.index = faiss.IndexFlatIP(768)
            self.id_index = faiss.IndexIDMap(self.index)

    def write_index_information(self, string):
        index_information_file = self.path + '/Index Information.txt'
        if os.path.exists(index_information_file):
            append_write = 'a'
        else:
            append_write = 'w'
        with open(index_information_file, append_write) as f:
            f.write(string + '\n')

    def restart(self, overall_time):
        with open(f'{self.path}/stats_before_failure.txt', 'a') as f:
            f.write(f'timing when failing at index {self.index_number}: {overall_time}\n')
        with open(f'{self.path}/failed list.txt', 'w') as f:
            f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in self.failed_list))
        # the arglist includes all arguments necessary to call this function and pick up where it left.
        # the first argument is overwritten by the system as it always holds the filename of the function
        # str(1) is for `read_from_file` as the datasset will never be created on resumption
        # index start will be the first index to generate embeddings for on resumption
        # index name start is
        arglist = ['filename', str(self.batch_size), self.name, self.data_location, str(1), self.path,
                   self.model,self.pooling, str(self.index_size), str(self.index_start), str(self.index_number),
                   str(self.fail_mode), str(self.batch_size), str(overall_time), f'{self.path}/failed list.txt']
        if not self.read_from_file:
            # the dataset will never be generated. but doesn't need to have that name
            arglist[3] = f'{self.name}_dataset'
        os.execv(__file__, arglist)


# if this script is called directly from the command line, it will be because generating faiss index failed.
# the script will automatically recover from a failure by calling itself again and starting where it had to leave off
# due to failure.
if __name__ == "__main__":
    print(sys.argv)
    if sys.argv[1] == 'start':
        if sys.argv[5] not in valid_averaging_modes:
            print('unknown pooling')
            exit(-1)
        if sys.argv[5] == 'SentenceBert':
            if sys.argv[6] not in valid_SentenceBert_model_names:
                print(f'{sys.argv[6]} is not a valid SentenceBert Model')
                exit(-1)
        elif sys.argv[6] not in valid_model_names:
            print(f'{sys.argv[6]} is not a valid Model for the pooling Strategy: {sys.argv[5]}')
            exit(-1)

        print(f'Starting Index creation')
        generator = IndexGenerator(int(sys.argv[2]), sys.argv[3], sys.argv[4], True,
                                   model=sys.argv[6], pooling=sys.argv[5])
    else:
        print(f'restarting the generation of the embeddings')
        generator = IndexGenerator(int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5],
                                   sys.argv[6], sys.argv[7], int(sys.argv[8]), int(sys.argv[9]),
                                   int(sys.argv[10]), int(sys.argv[11]), int(sys.argv[12]), float(sys.argv[13]),
                                   sys.argv[14])
    generator.generate_index()
