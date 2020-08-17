#!/usr/bin/env python

import os
import re
import shutil
import sys
import time
import csv
import statistics
from collections import Counter
from util import splitter
from tika_tests import read_in_documents
from elasticsearch_dsl import Index, connections, Search, MultiSearch
from elasticsearch_dsl.query import MoreLikeThis
from elastic import TestCase
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
from elasticsearch.helpers import bulk
from tabulate import tabulate
import faiss
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings
from tqdm import tqdm
from faiss_tests import search_on_disk
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


client = Elasticsearch()
result_path = './results'
qual_path = './qualitative analysis'
faiss_path = './faiss_indexes'
dataset_path = './datasets/generated'
INDEX_SIZE = 1000000


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

def get_document_by_Id(data_location, read_from_file, id):
    dataset = get_dataset(data_location, read_from_file)
    return dataset[id]




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
                                 minimum_should_match='5%',
                                 analyzer='stop'
                                 )
    else:
        like_list = [{'_index': index, '_id': item} for item in document_id]
        mlt_match = MoreLikeThis(fields=["content"],
                                 like=like_list,
                                 min_term_freq=1,
                                 min_doc_freq=1,
                                 analyzer='stop'
                                 )
    s = s.query(mlt_match)
    s = s[:size]
    return s


def get_mlt_results(document_id, index, size=20):
    s = create_mlt_with_id(document_id, index, size=size)
    response = s.execute()
    results = [document_id] + [f'{hit.meta.id} ({hit.meta.score})' for hit in response]
    return results


def es_create_result_csv_bulk(name, index, result_size=200, batch_size=1000):
    start_time = time.time()
    index_size = Search(index=index).count()
    rest = index_size % batch_size
    results = []
    for i in range(0, index_size - rest, batch_size):
        multisearch = MultiSearch(index=index)
        print(f'generating results number {i} to {i + batch_size}')
        for item in range(i, i + batch_size):
            multisearch = multisearch.add(create_mlt_with_id(item, index, result_size))
        responses = multisearch.execute()
        for index_id, response in enumerate(responses, start=i):
            results.append([str(index_id)] + [f'{hit.meta.id} ({hit.meta.score})' for hit in response])
    if rest:
        multisearch = MultiSearch(index=index)
        for i in range(index_size-rest, index_size):
            multisearch = multisearch.add(create_mlt_with_id(item, index, result_size))
        responses = multisearch.execute()
        for index_id, response in enumerate(responses, start=i):
            results.append([str(index_id)] + [f'{hit.meta.id} ({hit.meta.score})' for hit in response])
    try:
        os.mkdir(f'{faiss_path}/{name}/')
    except FileExistsError:
        print(f'directory already exists and I am just deleting it.')
        shutil.rmtree(f'{faiss_path}/{name}/')
        os.mkdir(f'{faiss_path}/{name}/')
    with open(f'{faiss_path}/{name}/search_rankings.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for line in results:
            wr.writerow(line)
    stop_time = time.time() - start_time
    with open(f'./datasets/elasticsearch_{name}_timing', 'a') as f:
        f.write(f'time for generating es results for {name}: {stop_time}\n')
    return stop_time

def es_create_result_csv(name, index, result_size=20):
    start_time = time.time()
    index_size = Search(index=index).count()
    es_results = [get_mlt_results(item, index, result_size) for item in
                  tqdm(range(index_size), desc=f'creating es results')]
    try:
        os.mkdir(f'{faiss_path}/{name}/')
    except FileExistsError:
        print(f'directory already exists and I am just deleting it.')
        shutil.rmtree(f'{faiss_path}/{name}/')
        os.mkdir(f'{faiss_path}/{name}/')
    with open(f'{faiss_path}/{name}/search_rankings.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for line in es_results:
            wr.writerow(line)
    stop_time = time.time() - start_time
    with open(f'./datasets/elasticsearch_{name}_timing', 'a') as f:
        f.write(f'time for generating es results for {name}: {stop_time}\n')
    return stop_time


def generate_embeddings(docs, batch_size, model_name='bert-base-cased', pooling='mean', offset=0):
    """
    Generator function for generating embeddings from strings using a flair model. Takes a list of sentences and
    returns a list tuple. The first element represents failure (0) or success (1 or 2) and
    the second element contains a list of embeddings as numpy arrays if successful, and the indices of the failed batch
    if unsuccessful.
    The first element is 1, if batch_size embeddings were created
    :param docs: a list of strings for which embeddings should be created
    :param batch_size: integer representing how many embeddings should be created at once
    :param model_name: the model for creating the embeddings. Defaults to document embeddings using BERT-Base
    :param pooling: the pooling strategy to generate Document Embeddings
    :param offset: the offset of the integers, for printing out the correct index
    :return: a tuple (success/failure, embeddings/failed_indices)
    """
    rest = len(docs) % batch_size
    model = False
    if pooling == 'mean':
        embedding = TransformerWordEmbeddings(model_name, layers='-1', allow_long_sentences=True)
        model = DocumentPoolEmbeddings([embedding], fine_tune_mode='none')
    elif pooling == 'CLS':
        model = TransformerDocumentEmbeddings(model_name)
    if model:
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
    elif pooling == 'SentenceBert':
        model = SentenceTransformer(model_name)
        for i in range(0, len(docs) - rest, batch_size):
            try:
                embeddings = model.encode(docs[i:i + batch_size])
                print(f'successfully embedded sentences {offset + i} to {offset + i + batch_size-1}')
                yield 1, embeddings
            except RuntimeError:
                print(f'could not embed sentences with index {offset + i} '
                      f'to {offset + i + batch_size-1}\nstoring in failed index list')
                yield 0, (offset + i, offset + i + batch_size-1)
        if rest:
            try:
                embeddings = model.encode(docs[-rest:])
                print(f'successfully embedded sentences from {len(docs) + offset - rest} to the end')
                yield 1, embeddings
            except RuntimeError:
                yield 0, (len(docs) - rest, 0)
    else:
        raise Exception("No Valid model")


def create_faiss_csv(name, batch_size=1000, k=100):
    start_time = time.time()
    index_files = []
    path = f'{faiss_path}/{name}'
    with open(f'{path}/Index Information.txt', 'r') as f:
        for line in f.readlines():
            line_splits = line.split(' ')
            if line_splits[0] == 'Index' and line_splits[2] == 'contains':
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
                                       [f"{entry} ({dists[index-start][entry_index]})"
                                        for entry_index, entry in enumerate(vector)])
    stop_time = time.time() - start_time
    with open(f'{path}/Index Information.txt', 'a') as f:
        f.write(f'time for generating faiss csv results for {name}: {stop_time}\n')


def index_generator(index, size, batch_size=1):
    rest = size % batch_size
    for i in range(0, size - rest, batch_size):
        yield index.reconstruct_n(i, batch_size)
    if rest:
        yield index.reconstruct_n(size-rest, rest)


def search_csv(csv_file, dataset, index, result_size=10):
    string_dataset = get_dataset(dataset, 1)
    text_results = [string_dataset[index]]
    table_results = [['queried index', string_dataset[index], 0]]
    with open(csv_file, 'r', newline='') as id_file:
        csv_reader = csv.reader(id_file, delimiter=',')
        for run_index, line in enumerate(csv_reader):
            if run_index == index:
                results = [split_rank(cell) for cell in line[2:]]
                ranks, values = zip(*results)
                text_results = [string_dataset[rank] for rank in (index,) + ranks[:result_size]]
                return text_results,  (index,) + ranks[:result_size]


def get_total_time(timing_file):
    total_time = 0
    with open(timing_file, 'r') as f:
        for line in f.readlines():
            test = re.split(': ', line)
            total_time += float(test[1])
    return total_time


def split_rank(cell):
    split_cell = cell.split('(')
    rank = int(split_cell[0])  # splitting at the parenthesis, the first string in the list is the rank
    # splitting at the parenthesis, the second string is the value but with an added
    # closing parenthesis
    score = float(split_cell[1][:-1])
    return rank, score  # returns a tuple of rank and score


def evaluate_rankings(name):
    with open(f'{faiss_path}/{name}/search_rankings.csv', 'r', newline='') as rankings_file:
        with open(f'{faiss_path}/{name}/search_evaluation.csv', 'w+', newline='') as results_file:
            csv_reader = csv.reader(rankings_file, delimiter=',')
            csv_writer = csv.writer(results_file)
            csv_writer.writerow(['search_id', 'target_pair_ranking', 'average_score_total', 'median_score_total',
                                'average_score_top_ten', 'median_score_top_ten', 'close_to_pair_hits_total',
                                 'close_to_pair_hits_top_ten', 'score_distance_total', 'score_distance_top_ten',
                                 'biggest_score_drop'])
            for row in csv_reader:
                search_id = split_rank(row[1])[0]
                if search_id % 2:  # if the id is odd, the target ID is the one below
                    target_pair_id = search_id - 1
                else:  # otherwise it is the one above
                    target_pair_id = search_id + 1
                # convert all results into id and score
                results = [split_rank(cell) for cell in row[2:]]
                ranks, values = zip(*results)
                if target_pair_id in ranks:
                    target_pair_ranking = ranks.index(target_pair_id) + 1
                else:
                    target_pair_ranking = 201
                average_score_total = statistics.mean(values)
                median_score_total = statistics.median(values)
                average_score_top_ten = statistics.mean(values[:10])
                median_score_top_ten = statistics.median(values[:10])
                close_to_pair_start = target_pair_id - target_pair_id % 2 - 10
                #  these are the indices of the 10 closest pairs over/under the evaluated pair
                close_to_pair_range = [i for i in range(close_to_pair_start, close_to_pair_start + 10)] +\
                                      [i for i in range(close_to_pair_start + 12, close_to_pair_start + 22)]
                close_to_pair_hits_total = sum(value in ranks for value in close_to_pair_range)
                close_to_pair_hits_top_ten = sum(value in ranks for value in close_to_pair_range[:10])
                score_distance_total = values[0] - values[-1]
                if len(values) >= 10:
                    score_distance_top_ten = values[0] - values[9]
                else:
                    score_distance_top_ten = score_distance_total
                biggest_score_drop = max(abs(element - values[index-1])
                                         for index, element in enumerate(values[1:], start=1))
                row_results = [search_id, target_pair_ranking, average_score_total, median_score_total,
                               average_score_top_ten, median_score_top_ten, close_to_pair_hits_total,
                               close_to_pair_hits_top_ten, score_distance_total, score_distance_top_ten,
                               biggest_score_drop]
                csv_writer.writerow(row_results)
    results_spreadsheet = pd.read_csv(f'{faiss_path}/{name}/search_evaluation.csv')
    description = results_spreadsheet.describe()
    description.to_csv(f'{faiss_path}/{name}/results_description.csv')
    rank_counts = results_spreadsheet.groupby('target_pair_ranking').count()['search_id']
    rank_counts.to_csv(f'{faiss_path}/{name}/rank_counts.csv')
    no_finders = results_spreadsheet[results_spreadsheet.target_pair_ranking == 201]
    no_finders.to_csv(f'{faiss_path}/{name}/no_finders.csv')
    no_finders_description = no_finders.describe()
    no_finders_description.to_csv(f'{faiss_path}/{name}/no_finders_description.csv')
    finders = results_spreadsheet[results_spreadsheet.target_pair_ranking != 201]
    finders.to_csv(f'{faiss_path}/{name}/finders.csv')
    finders_description = finders.describe()
    finders_description.to_csv(f'{faiss_path}/{name}/finders_description.csv')


def compare_models(dataset_name):
    plt.rcParams.update({'axes.titlesize': 'large'})
    with open(f'{faiss_path}/master_thesis_results/{dataset_name}_es/search_rankings.csv', 'r') as es_data:
        csv_reader = csv.reader(es_data, delimiter=',')
        all_es_ranks = []
        for row in csv_reader:
            results = [split_rank(cell) for cell in row[1:]]
            ranks, values = zip(*results)
            all_es_ranks.append(ranks)
    for filename in os.listdir(f'{faiss_path}/master_thesis_results'):
        model_name = filename.split('_')[-1]
        if filename.startswith(dataset_name):
            co_occurence = []
            top_10_from_es_count = []
            top_10_from_model_count = []
            no_finders = 0
            with open(f'{faiss_path}/master_thesis_results/{filename}/search_rankings.csv', 'r') as model_data:
                csv_reader = csv.reader(model_data, delimiter=',')
                for index, row in tqdm(enumerate(csv_reader), desc=f'creating co occurences for {filename}'):
                    results = [split_rank(cell) for cell in row[2:]]
                    ranks, values = zip(*results)
                    co_occurence.append(sum(all_es_ranks[index].count(item) for item in ranks))
                    top_10_from_es_count.append(sum(ranks.count(item) for item in all_es_ranks[index][:10]))
                    top_10_from_model_count.append(sum(all_es_ranks[index].count(item) for item in ranks[:10]))
                all_counts = Counter(co_occurence)
                average_co_occurrence = statistics.mean(co_occurence)
                average_top_10_from_es = statistics.mean(top_10_from_es_count)
                average_top_10_from_model = statistics.mean(top_10_from_model_count)
                std_es = statistics.stdev(top_10_from_es_count)
                std_model = statistics.stdev(top_10_from_model_count)
                standard_deviation = statistics.stdev(co_occurence)
                plt.grid(color='gray', linestyle='dashed')
                plt.xlim(0, 145)
                plt.bar(all_counts.keys(), all_counts.values(), 1.0, color='b')
                plt.rc('xtick', labelsize=8)
                plt.rc('ytick', labelsize=8)
                plt.xlabel('number of occurrences [-]', fontsize=20)
                plt.ylabel('co-occurrences [-]', fontsize=20)
                plt.title(f' {model_name} ', fontsize=20)

                plt.axvline(average_co_occurrence, color='k', linestyle='dashed', linewidth=1)
                min_ylim, max_ylim = plt.ylim()
                plt.text(average_co_occurrence * 1.1, max_ylim * 0.9,
                         f'Mean: {statistics.mean(co_occurence):.2f}')
                plt.savefig(f'{faiss_path}/master_thesis_results/{filename}/'
                            f'{filename[len(dataset_name)+1:]}_co_occurrence_plot.pdf',
                            bbox_inches='tight')
                plt.clf()
                # find the indices of the five documents with highest and lowest co-occurences each.
                top_5_indices = sorted(range(len(co_occurence)), key=lambda i: co_occurence[i])[-5:]
                bottom_5_indices = sorted(range(len(co_occurence)), key=lambda i: co_occurence[i])[:5]
                top_5_with_values = [(index, co_occurence[index]) for index in top_5_indices]
                bottom_5_with_values = [(index, co_occurence[index]) for index in bottom_5_indices]
            with open(f'{faiss_path}/master_thesis_results/{filename}/co-occurence.csv', 'w') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Average Co-occurrence', average_co_occurrence])
                csv_writer.writerow(['standard_deviation', standard_deviation])
                csv_writer.writerow(['top five indices (and Values)'] +
                                    [f'{item[0]} ({item[1]})' for item in top_5_with_values])
                csv_writer.writerow(['bottom five (and Values)'] +
                                    [f'{item[0]} ({item[1]})' for item in bottom_5_with_values])
                csv_writer.writerow(['Average top ten co-occurrence in elastic', average_top_10_from_model])
                csv_writer.writerow(['standard_deviation top 10 from model in es', std_model])
                csv_writer.writerow(['Average top ten co-occurrence in model', average_top_10_from_es])
                csv_writer.writerow(['standard_deviation top ten from es in model', std_es])


def compare_models_top_ten(dataset_name):
    with open(f'{faiss_path}/master_thesis_results/{dataset_name}_es/search_rankings.csv', 'r') as es_data:
        csv_reader = csv.reader(es_data, delimiter=',')
        all_es_ranks = []
        for row in csv_reader:
            results = [split_rank(cell) for cell in row[1:11]]
            ranks, values = zip(*results)
            all_es_ranks.append(ranks)
            not_found =[]
    for filename in os.listdir(f'{faiss_path}/master_thesis_results'):
        model_name = filename.split('_')[-1]
        if filename.startswith(dataset_name):
            co_occurence = []
            with open(f'{faiss_path}/master_thesis_results/{filename}/search_rankings.csv', 'r') as model_data:
                csv_reader = csv.reader(model_data, delimiter=',')
                for index, row in tqdm(enumerate(csv_reader), desc=f'creating co occurences for {filename}'):
                    results = [split_rank(cell) for cell in row[2:12]]
                    ranks, values = zip(*results)
                    co_oc_count = sum(all_es_ranks[index].count(item) for item in ranks)
                    co_occurence.append(co_oc_count)
                    if not co_oc_count:
                        not_found.append(index)
                all_counts = Counter(co_occurence)
                average_co_occurrence = statistics.mean(co_occurence)
                standard_deviation = statistics.stdev(co_occurence)
                plt.grid(color='gray', linestyle='dashed')
                plt.xlim(0, 10)
                plt.bar(all_counts.keys(), all_counts.values(), 1.0, color='b')
                plt.rc('xtick', labelsize=8)
                plt.rc('ytick', labelsize=8)
                plt.xlabel('number of occurrences [-]', fontsize=20)
                plt.ylabel('co-occurrences [-]', fontsize=20)
                plt.title(f' {model_name} ', fontsize=20)

                plt.axvline(average_co_occurrence, color='k', linestyle='dashed', linewidth=1)
                min_ylim, max_ylim = plt.ylim()
                plt.text(average_co_occurrence * 1.1, max_ylim * 0.9,
                         f'Mean: {statistics.mean(co_occurence):.2f}')
                plt.savefig(f'{faiss_path}/master_thesis_results/{filename}/'
                            f'{filename[len(dataset_name)+1:]}_co_occurrence_plot_10.pdf',
                            bbox_inches='tight')
                plt.clf()
                # find the indices of the five documents with highest and lowest co-occurences each.
                top_5_indices = sorted(range(len(co_occurence)), key=lambda i: co_occurence[i])[-5:]
                bottom_5_indices = sorted(range(len(co_occurence)), key=lambda i: co_occurence[i])[:5]
                top_5_with_values = [(index, co_occurence[index]) for index in top_5_indices]
                bottom_5_with_values = [(index, co_occurence[index]) for index in bottom_5_indices]
            with open(f'{faiss_path}/master_thesis_results/{filename}/co-occurence_10.csv', 'w') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Average Co-occurrence', average_co_occurrence])
                csv_writer.writerow(['standard_deviation', standard_deviation])
                csv_writer.writerow(['top five indices (and Values)'] +
                                    [f'{item[0]} ({item[1]})' for item in top_5_with_values])
                csv_writer.writerow(['bottom five (and Values)'] +
                                    [f'{item[0]} ({item[1]})' for item in bottom_5_with_values])
            with open(f'{faiss_path}/master_thesis_results/{filename}/no_co-oc_in_top_10', 'w') as not_found_file:
                for no_finder in not_found:
                    not_found_file.write(f'{no_finder}\n')




def get_ranking_data(dataset_name):
    pair_rankings = []
    for filename in os.listdir(f'{faiss_path}/master_thesis_results'):
        if filename.startswith(dataset_name):
            model_data = pd.read_csv(f'{faiss_path}/master_thesis_results/{filename}/finders.csv')
            pair_rankings.append([filename, model_data['target_pair_ranking']])
    return pair_rankings


def create_qual_analysis(dataset_name, id, top_x=10):
    save_path = f'{qual_path}/{dataset_name}_{id}'
    try:
        os.mkdir(save_path)
    except FileExistsError:
        print(f'directory already exists and I am just deleting it.')
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    dataset = get_dataset(dataset_name, 1)
    for filename in os.listdir(f'{faiss_path}/master_thesis_results'):
        if filename.startswith(dataset_name):
            query_document = dataset[id]
            if id % 2:
                target_document_id = id - 1
            else:
                target_document_id = id + 1
            target_document = dataset[target_document_id]
            str_results, ranks = search_csv(
                f'{faiss_path}/master_thesis_results/{filename}/search_rankings.csv',
                dataset_name, id, top_x)
            with open(f'{save_path}/{filename[len(dataset_name)+1:]}.txt', 'w') as textfile:
                textfile.write(f'-- analysis from {filename} -- \n')
                textfile.write(f'-- Query Document ({id}) -- \n')
                textfile.write(query_document + '\n\n')
                textfile.write(f'-- Target Document ({target_document_id}) -- \n')
                textfile.write(target_document + '\n\n')
                textfile.write(f'-- Ranking results (with ID) -- \n')
                for rank, text in enumerate(str_results[1:], start=1):
                    textfile.write(f'{rank}: ({ranks[rank]}):\n')
                    textfile.write(text + '\n\n')








if __name__ == "__main__":
    print(f'generating results for {sys.argv[1]}')
    create_faiss_csv(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    print(f'Evaluating results for {sys.argv[1]}')
    evaluate_rankings(sys.argv[1])
    print(f'moving everything to the external drive')
    shutil.move(f'{faiss_path}/{sys.argv[1]}', f'{faiss_path}/master_thesis_results')





