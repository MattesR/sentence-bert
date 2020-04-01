#!/usr/bin/env python

import os
import sys
import time
from util import splitter
from tika_tests import read_in_documents
from elasticsearch_dsl import Index, connections, Search
from elasticsearch_dsl.query import MoreLikeThis
from elastic import TestCase
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import bulk
import itertools
import faiss
import numpy as np
import flair_tests
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings
import torch
import shutil

client = Elasticsearch()
result_path = './results'
faiss_path = './faiss_indexes'
dataset_path = './datasets/generated'
INDEX_SIZE = 200000
embedding = BertEmbeddings(layers='-1')
document_embeddings = DocumentPoolEmbeddings([embedding], fine_tune_mode='nonlinear')


def create_test_dataset(path, page_size, write_name=False):
    docs = read_in_documents(path)
    pairs = [splitter.create_document_pairs(splitter.create_pseudo_pages(document, page_size)) for document in docs]
    flat_pairs = [paragraph for document in pairs for paragraph in document]
    flat_pairs = [item.strip() for item in flat_pairs]  # stripping here to make sure that exactly the same is loaded
    if write_name is not False:
        print(f'writing index {write_name} to disk')
        with open(dataset_path + f'/{write_name}', 'w') as f:
            f.write('\n'.join(flat_pairs))
    return flat_pairs


def get_dataset(data_location, read_from_file, page_size=4, write=False):
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
            flat_pairs = [line.strip() for line in f]
    else:
        flat_pairs = create_test_dataset(data_location, page_size, write_name=write)
    return flat_pairs


def delete_es_index(name):
    new_index = Index(name)
    try:
        response = new_index.delete()
        if response['acknowledged']:
            print(f'delete index {name}')
        else:
            print(f'deletion not acked, instead it returned {response}')
    except NotFoundError:
        print(f'the index with name {name} was not found')


def create_new_es_testindex(name):
    connection = connections.create_connection(hosts=['localhost'], timeout=20)
    new_index = Index(name)
    new_index.document(TestCase)
    new_index.create()


def bulk_add_collection_to_es(path, name, page_size=4, min_size=0):
    create_new_es_testindex(name)
    connections.create_connection(hosts=['localhost'], timeout=20)
    documents = read_in_documents(path, min_size=min_size)
    print(f'loaded all documents from {path}')
    pairs = [splitter.create_document_pairs(splitter.create_pseudo_pages(document, page_size)) for document in documents]
    es_docs = [TestCase(meta={'id': paragraph_id}, content=pseudo_paragraph) for paragraph_id, pseudo_paragraph
               in enumerate(itertools.chain(*pairs))]
    TestCase.init()
    print(f'bulk adding {len(es_docs)} documents to es index {name}')
    bulk(connections.get_connection(), (d.to_dict(True) for d in es_docs))
    return len(es_docs)


def create_mlt_with_id(document_id, index, size=20):
    s = Search(using=client, index=index)
    if not isinstance(document_id, list):
        mlt_match = MoreLikeThis(fields=["content"],
                                 like={'_index': index, '_id': document_id},
                                 min_term_freq=1,
                                 min_doc_freq=1)
    else:
        like_list = [{'_index': index, '_id': item} for item in document_id]
        mlt_match = MoreLikeThis(fields=["content"],
                                 like=like_list,
                                 min_term_freq=1,
                                 )

    s = s.query(mlt_match)
    s = s[:size]
    return s


def get_mlt_results(document_id, index, size=20):
    s = create_mlt_with_id(document_id, index, size=20)
    response = s.execute()
    result_ids = [int(hit.meta.id) for hit in response]
    result_values = [int(hit.meta.score) for hit in response]
    return result_ids, result_values


def generate_faiss_index(batch_size, name, data_location, read_from_file, path=faiss_path,
                         index_size=INDEX_SIZE, index_start=0, index_name_start=0,  failed_list=None):
    document_pairs = get_dataset(data_location, read_from_file, write=f'{name}_dataset')
    index = faiss.IndexFlatL2(768)
    id_index = faiss.IndexIDMap(index)
    generator = generate_embeddings(document_pairs[index_start:], batch_size, offset=index_start)
    if not failed_list:
        failed_list = []
    else:
        with open(failed_list, 'r') as f:
            failed_list = [tuple(map(int, line.split(' '))) for line in f]
    start_time = time.time()
    if len(document_pairs[index_start:]) <= index_size:
        index_position = index_start
        for embedding_number, embeddings in enumerate(generator, start=index_start):
            index_position += batch_size
            if embeddings[0]:
                id_index.add_with_ids(flair_tests.array_from_list(embeddings[1]),
                                      np.asarray([j for j in range(index_position,
                                                                   index_position + batch_size)])
                                      )
            else:
                print(f'failed at index {index_position} to {index_position + batch_size}, restarting after that batch')
                failed_list.append(embeddings[1])
                faiss.write_index(id_index, path + f'/{name}_interrupt_{index_name_start}')
                index_start = embeddings[1][1] + 1
                print(f'index start is at {index_start}')
                arglist = ['filename', str(batch_size), name, data_location, str(read_from_file), path,
                           str(index_size), str(index_start), str(index_name_start),  failed_list]
                overall_time = time.time() - start_time
                wrap_up(arglist, failed_list, embedding_number, batch_size, name, index_name_start, overall_time)
                break

        faiss.write_index(index, path + f'/{name}')
        print(f'saved one faiss index in file with name {name}')
    else:
        current_index_size = 0
        index_position = index_start
        index_number = index_name_start
        path += f'/{name}'
        if index_number == 0:
            try:
                os.mkdir(path)
            except FileExistsError:
                print(f'directory already exists and I am just deleting it. Call the Police, I do not care')
                shutil.rmtree(path)
                os.mkdir(path)
        for embedding_number, embeddings in enumerate(generator, start=index_start):
            current_index_size += batch_size
            index_position += batch_size
            if embeddings[0]:
                if current_index_size < index_size:
                    id_index.add_with_ids(flair_tests.array_from_list(embeddings[1]),
                                          np.asarray([j for j in range(index_position,
                                                                       index_position + batch_size)])
                                          )
                else:
                    faiss.write_index(id_index, path + f'/{name}_{index_number}')
                    index = faiss.IndexFlatL2(768)
                    index_number += 1
                    current_index_size = batch_size
                    id_index = faiss.IndexIDMap(index)
                    id_index.add_with_ids(flair_tests.array_from_list(embeddings[1]),
                                          np.asarray([j for j in range(index_position,
                                                                       index_position + batch_size)])
                                          )
            else:
                print(f'failed at index {index_position} to {index_position + batch_size}, restarting after that batch')
                failed_list.append(embeddings[1])
                index_start = embeddings[1][1] + 1
                print(f'index start is at {index_start}')
                faiss.write_index(id_index, path + f'/{name}_interrupt_{index_name_start}')
                overall_time = time.time() - start_time
                arglist = ['filename', str(batch_size), name, data_location, str(read_from_file), path,
                           str(index_size), str(index_start), str(index_name_start),  failed_list]
                wrap_up(arglist, failed_list, name, index_name_start, overall_time)

        print(f'saved {index_number} indices in directory with name {name}')
    if failed_list:
        return failed_list
    else:
        return True


def generate_embeddings(docs, batch_size, model=document_embeddings, offset=0):
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
            print(f'successfully embedded sentences from {-rest} to the end')
            yield 1, [sentence.get_embedding().detach().cpu().numpy() for sentence in sentences]
        except RuntimeError:
            yield 0, (-rest, 0)
        yield [sentence.get_embedding().detach().cpu().numpy() for sentence in sentences]


def wrap_up(old_arguments, failed_list, name, index_name_start, overall_time):
    with open(f'{dataset_path}/{name}_stats_before_failure', 'a') as f:
        f.write(f'timing when failing at index {index_name_start}: {overall_time}\n')
    with open(f'{dataset_path}/{name}_failed list.txt', 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in failed_list))
    # the arglist includes all arguments necessary to call this function and pick up where it left.
    # the first argument is overwritten by the system as it always holds the filename of the function
    # str(1) is for `read_from_file` as the datasset will never be created on resumption
    # index start will be the first index to generate embeddings for on resumption
    # index name start is
    arglist = old_arguments
    arglist[3] = f'{name}_dataset'  # the dataset will never be generated
    arglist[4] = str(1)
    arglist[8] = str(index_name_start + 1)
    arglist[9] = dataset_path + f'/{name}_failed list.txt'  # the failed list will be loaded
    os.execv(__file__, arglist)


# if this script is called directly from the command line, it will be because generating faiss index failed.
# the script will automatically recover from a failure by calling itself again and starting where it had to leave off
# due to failure.
if __name__ == "__main__":
    print(f'restarting the generation of the embeddings')
    arglist = sys.argv
    generate_faiss_index(int(arglist[1]), arglist[2], arglist[3], int(arglist[4]), arglist[5],
                         int(arglist[6]), int(arglist[7]), int(arglist[8]), arglist[9])
