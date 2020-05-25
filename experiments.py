#!/usr/bin/env python

import os
import sys
import re
import time
import csv
from util import splitter
from tika_tests import read_in_documents
from elasticsearch_dsl import Index, connections, Search
from elasticsearch_dsl.query import MoreLikeThis
from elastic import TestCase
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
from elasticsearch.helpers import bulk
from tabulate import tabulate
import faiss
import numpy as np
import flair_tests
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings
import shutil
from tqdm import tqdm
from faiss_tests import search_on_disk

client = Elasticsearch()
result_path = './results'
faiss_path = './faiss_indexes'
dataset_path = './datasets/generated'
INDEX_SIZE = 1000000
embedding = BertEmbeddings(layers='-1')
document_embeddings = DocumentPoolEmbeddings([embedding], fine_tune_mode='none')

csv_result_header = ['ranking', 'paragraph', 'distance']


def create_test_dataset(path, page_size=4, write_name=''):
    docs = read_in_documents(path)
    pairs = [splitter.create_document_pairs(splitter.create_pseudo_pages(document, page_size)) for document in docs]
    flat_pairs = [paragraph for document in pairs for paragraph in document]
    flat_pairs = [item.strip() for item in flat_pairs]  # stripping here to make sure that exactly the same is loaded
    if write_name is not False:
        print(f'writing index {write_name} to disk')
        with open(dataset_path + f'/{write_name}', 'w') as f:
            f.write('\n'.join(flat_pairs))
    return flat_pairs


def get_dataset(data_location, read_from_file, page_size=4, write=''):
    """
    gets the dataset according to the input. If the read_from_file Flag is set, the function loads and returns the
    generated dataset with the name, which is given as the input.
    if the read_from_file Flag is set to False, the dataset will be created. Input is then expected to be a path.
    The write flag makes it possible to write the dataset as a generated dataset to disk and expects a string for a
    name.
    :param data_location: either a name to load or a path to create a dataset
    :param page_size: number of paragraphs which make up a pseudo page
    :param read_from_file: TRUE: load dataset with name from input, FALSE: generate dataset with path from input
    :param write: name with which to store the generated dataset if desired.
    :return: the dataset as a flat list of strings, which are document pairs for comparision
    """
    if read_from_file:
        with open(dataset_path + f'/{data_location}', 'r') as f:
            flat_pairs = [line.strip() for line in f]  # primarily for filtering out empty lines at the end
            filter_object = filter(lambda x: x != "", flat_pairs)
            flat_pairs = list(filter_object)
    else:
        flat_pairs = create_test_dataset(data_location, page_size, write_name=write)
    return flat_pairs


def delete_es_index(name):
    new_index = Index(name)
    try:
        response = new_index.delete()
        if response['acknowledged']:
            print(f'deleted index {name}')
        else:
            print(f'deletion not acked, instead it returned {response}')
    except NotFoundError:
        print(f'the index with name {name} was not found')


def create_new_es_testindex(name):
    new_index = Index(name)
    new_index.document(TestCase)
    try:
        new_index.create()
        return True
    except RequestError:
        print(f'the index with name {name} already exists, delete first.')
        return False


def bulk_add_collection_to_es(data_location, read_from_file, name):
    document_pairs = get_dataset(data_location, read_from_file, write=f'{name}_dataset')
    start_time = time.time()
    if not create_new_es_testindex(name):
        delete_es_index(name)
        create_new_es_testindex(name)
    connections.create_connection(hosts=['localhost'], timeout=20)
    es_docs = [TestCase(meta={'id': paragraph_id}, content=pseudo_paragraph)
               for paragraph_id, pseudo_paragraph in enumerate(document_pairs)]
    TestCase.init()
    print(f'bulk adding {len(es_docs)} documents to es index {name}')
    bulk(connections.get_connection(), (d.to_dict(True) for d in es_docs))
    stop_time = time.time() - start_time
    with open(f'./datasets/elasticsearch_{name}_timing', 'w') as f:
        f.write(f'time for creating es index {name}: {stop_time}\n')
        f.write(f'number of items in the index: {len(es_docs)}')
    return len(es_docs)


def create_mlt_with_id(document_id, index, size=20):
    s = Search(using=client, index=index)
    if not isinstance(document_id, list):
        mlt_match = MoreLikeThis(fields=["content"],
                                 like={'_index': index, '_id': document_id},
                                 min_term_freq=1,
                                 min_doc_freq=1,
                                 minimum_should_match='5%')
    else:
        like_list = [{'_index': index, '_id': item} for item in document_id]
        mlt_match = MoreLikeThis(fields=["content"],
                                 like=like_list,
                                 min_term_freq=1,
                                 min_doc_freq=1,

                                 )
    s = s.query(mlt_match)
    s = s[:size]
    return s


def get_mlt_results(document_id, index, size=20):
    s = create_mlt_with_id(document_id, index, size=size)
    response = s.execute()
    result_ids = [int(hit.meta.id) for hit in response]
    result_values = [int(hit.meta.score) for hit in response]
    return result_ids, result_values


def es_create_result_csv(name, index_size, index, result_size=20):
    start_time = time.time()
    es_results = [get_mlt_results(item, index, result_size) for item in
                  tqdm(range(index_size), desc=f'creating es results')]
    with open(f'./{name}_es_results.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for line in es_results:
            wr.writerow(line)
    rankings = [item[0].split() for item in es_results]
    stop_time = time.time() - start_time
    short_indices = [index for index, ranking in enumerate(rankings) if len(ranking) < result_size]
    with open(f'./datasets/elasticsearch_{name}_timing', 'a') as f:
        f.write(f'time for generating es results for {name}: {stop_time}\n')
    return stop_time


def generate_embeddings(docs, batch_size, model=document_embeddings, offset=0):
    """
    Generator function for generating embeddings from strings using a flair model. Takes a list of sentences and
    returns a list tuple. The first element represents failure (0) or success (1 or 2) and
    the second element contains a list of embeddings as numpy arrays if successful, and the indices of the failed batch
    if unsuccessful.
    The first element is 1, if batch_size embeddings were created
    :param docs: a list of strings for which embeddings should be created
    :param batch_size: integer representing how many embeddings should be created at once
    :param model: the model for creating the embeddings. Defaults to document embeddings using BERT-Base
    :param offset: the offset of the integers, for printing out the correct index
    :return: a tuple (success/failure, embeddings/failed_indices)
    """
    rest = len(docs) % batch_size
    for i in range(0, len(docs) - rest, batch_size):
        sentences = [Sentence(sentence)for sentence in docs[i:i + batch_size]]
        try:
            model.embed(sentences)
            print(f'successfully embedded sentences {offset + i} to {offset + i + batch_size-1}')
            yield 1, [sentence.get_embedding().detach().cpu().numpy() for sentence in sentences]
        except RuntimeError:
            print(f'could not embed sentences with index {offset + i} '
                  f'to {offset + i + batch_size-1}\nstoring in failed index list')
            yield 0, (offset + i, offset + i + batch_size-1)
    if rest:
        sentences = [Sentence(sentence)for sentence in docs[-rest:]]
        try:
            model.embed(sentences)
            print(f'successfully embedded sentences from {len(docs) + offset - rest} to the end')
            yield 1, [sentence.get_embedding().detach().cpu().numpy() for sentence in sentences]
        except RuntimeError:
            yield 0, (len(docs) - rest, 0)


def create_faiss_csv(name, batch_size=1000, k=100):
    start_time = time.time()
    index_files = []
    path = f'{faiss_path}/{name}'
    with open(f'{path}/Index Information.txt', 'r') as f:
        for line in f.readlines():
            line_splits = line.split(' ')
            if line_splits[0] == 'Index':
                index_files.append({'name': line_splits[1],
                                    'start': int(line_splits[-3]),
                                    'end': int(line_splits[-1].rstrip())}
                                   )
    with open(f'{path}/search_rankings.csv', 'w', newline='') as id_file:
        wr_id = csv.writer(id_file, quoting=csv.QUOTE_ALL)
        for filename in os.listdir(f'{path}/'):
            if any(i['name'] == filename for i in index_files):
                id_index = faiss.read_index(f'{path}/{filename}')
                size = id_index.id_map.size()
                start = id_index.id_map.at(0)
                if size > batch_size:
                    generator = index_generator(id_index.index, size, batch_size)
                    for batch_number, batch_set in tqdm(enumerate(generator), total=size // batch_size,
                                                         desc=f'evaluating embeddings in {filename}'):
                        ids, dists = search_on_disk(path, batch_set, k+1)
                        batch_index_start = batch_number * batch_size + start
                        for index, vector in enumerate(ids):
                            wr_id.writerow([batch_index_start + index] +
                                           [f"{entry} ({dists[index][entry_index]})"
                                            for entry_index, entry in enumerate(vector)])
                else:
                    vectors = id_index.index.reconstruct_n(0, size)
                    ids, dists = search_on_disk(path, vectors, k+1)
                    for index, vector in enumerate(ids, start=start):
                        wr_id.writerow([str(index)] +
                                       [f"{entry} ({dists[index-start]})" for entry in vector])
    stop_time = time.time() - start_time
    with open(f'{path}/Index Information.txt', 'a') as f:
        f.write(f'time for generating faiss csv results for {name}: {stop_time}\n')


def index_generator(index, size, batch_size=1):
    rest = size % batch_size
    for i in range(0, size - rest, batch_size):
        yield index.reconstruct_n(i, batch_size)
    if rest:
        yield index.reconstruct_n(size-rest, rest)


def search_csv(csv_file, dataset, index, result_size):
    string_dataset = get_dataset(dataset, 1)
    table_results = [['queried index', string_dataset[index], 0]]
    with open(csv_file, 'r', newline='') as id_file:
        csv_reader = csv.reader(id_file, delimiter=',')
        for line in csv_reader:
            if int(line[0]) == index:
                ranking_indexes = [int(cell.split(' ')[0]) for cell in line[1:]]
                distances = [float(cell.split(' ')[1].strip('()')) for cell in line[1:result_size]]
                for ranking_number, ranking_index in enumerate(ranking_indexes, start=1):
                    table_results.append([ranking_number, string_dataset[ranking_index], distances[ranking_number-1]])
    print(tabulate(table_results, headers=csv_result_header))


def get_total_time(timing_file):
    total_time = 0
    with open(timing_file, 'r') as f:
        for line in f.readlines():
            test = re.split(': ', line)
            total_time += float(test[1])
    return total_time
