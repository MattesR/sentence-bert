import os
from util.splitter import flair_pairs_from_strings
from tika_tests import read_in_documents

def get_results(path, model):
    documents = read_in_documents(path)
    pair_list = [flair_pairs_from_strings(document, 4) for document in documents]


def create_batch(list,size):
