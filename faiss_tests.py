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
headers = ['unit_id', 'content', 'position', 'document_id']
from util import database as db
import time


def similarity_from_string(string, k, model=model, faiss_path=faiss_path):
    t_start = time.time()
    faiss_embedding = string_to_faiss_embedding(model, string)
    similarity_ids = search_on_disk(faiss_path, faiss_embedding, k)
    with psycopg2.connect(host='localhost', database=DBNAME, user=DBUSER, password=DBPASSWORD) as con:
        results = db.get_sentences_from_ids(con, similarity_ids)
    t_stop = time.time()
    print(tabulate(results, headers=headers))
    print (f' Getting these results took {t_stop - t_start} seconds')



def search_on_disk(path, embedding, k):
    """
    returns a tuple of ids to look up in the postgres database
    """
    i = 0
    result_heap = ResultHeap(nq=len(embedding), k=k)
    for filename in os.listdir(path):
        if filename != '.gitignore':
            index = faiss.read_index(faiss_path + '/' + filename)
            dist, ids = index.search(embedding, k)
            result_heap.add_batch_result(dist, ids, i * index_size)
    result_heap.finalize()
    tuple_ids = tuple(map(tuple, result_heap.I))
    if len(embedding == 1):
        tuple_ids = tuple_ids[0]
    return tuple(id.item() for id in tuple_ids)




