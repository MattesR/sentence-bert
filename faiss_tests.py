import os
import time
from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from util.embeddings import ResultHeap
from util.embeddings import string_to_faiss_embedding
import psycopg2

result_path = './results'
faiss_path = './faiss_indexes'
model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)
DBNAME = 'enron'
DBUSER = 'postgres'
DBPASSWORD = 'postgres'
ENRON_MAIL_PATH = "../datasets/enron"
test_path = "/home/moiddes/opt/datasets/enron/white-s/val"
index_size = 200000
tabulate_headers = ['unit_id', 'content', 'position', 'document_id']
from util import database as db
import time
from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings

embedding = BertEmbeddings(layers='-1')
document_embeddings = DocumentPoolEmbeddings([embedding], fine_tune_mode='nonlinear')



def similarity_from_string(string, k, model=model, faiss_path=faiss_path):
    t_start = time.time()
    faiss_embedding = string_to_faiss_embedding(model, string)
    similarity_ids = search_on_disk(faiss_path, faiss_embedding, k)
    with psycopg2.connect(host='localhost', database=DBNAME, user=DBUSER, password=DBPASSWORD) as con:
        results = db.get_sentences_from_ids(con, similarity_ids)
    t_stop = time.time()
    print(tabulate(results, headers=tabulate_headers))
    print(f' Getting these results took {t_stop - t_start} seconds')


def get_ranking(string, k, model=model, faiss_path=faiss_path):
    t_start = time.time()
    faiss_embedding = string_to_faiss_embedding(model, string)
    similarity_ids = search_on_disk(faiss_path, faiss_embedding, k)
    t_stop = time.time()
    print(f' Getting these results took {t_stop - t_start} seconds')
    return similarity_ids


def search_on_disk(path, embedding, k):
    """
    returns a tuple of ids to look up in the postgres database
    The path must contain a File called Index Information.txt, which stores the name and start and end points for
    the index files.
    """
    index_files = []
    result_heap = ResultHeap(nq=len(embedding), k=k)
    with open(f'{path}/Index Information.txt', 'r') as f:
        for line in f.readlines():
            linesplits = line.split(' ')
            if linesplits[0] == 'Index':
                index_files.append({'name': linesplits[1],
                                    'start': int(linesplits[-3]),
                                    'end': int(linesplits[-1].rstrip())}
                                   )
    for filename in os.listdir(f'{path}/'):
        if any(i['name'] == filename for i in index_files):
            print(f'adding results from {filename}')
            id_index = faiss.read_index(f'{path}/{filename}')
            start = id_index.id_map.at(0)
            dist, ids = id_index.search(embedding, k)
            result_heap.add_batch_result(dist, ids, start)
    result_heap.finalize()

    return result_heap.I, result_heap.D





