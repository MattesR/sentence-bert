import os
from util import splitter
from tika_tests import read_in_documents
from elasticsearch_dsl import Index, connections, Search
from elasticsearch_dsl.query import MoreLikeThis
from elastic import TestCase
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import itertools
client = Elasticsearch()


def get_results(path, model):
    documents = read_in_documents(path)
    pair_list = [splitter.flair_pairs_from_strings(document, 4) for document in documents]


def create_batch(items, size):
    pass


def create_new_es_testindex(name):
    connection = connections.create_connection(hosts=['localhost'], timeout=20)
    new_index = Index(name)
    new_index.document(TestCase)
    new_index.create()


def bulk_add_collection_to_es(path, name, textUnits=4):
    create_new_es_testindex(name)
    connections.create_connection(hosts=['localhost'], timeout=20)
    documents = read_in_documents(path)
    print(f'loaded all documents from {path}')
    pairs = [splitter.create_document_pairs(splitter.create_pseudo_pages(document, textUnits)) for document in documents]
    es_docs = [TestCase(meta={'id': paragraph_id}, content=pseudo_paragraph) for paragraph_id, pseudo_paragraph
               in enumerate(itertools.chain(*pairs))]
    TestCase.init()
    print(f'bulk adding all {len(es_docs)} documents to es index {name}')
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
    result_ids = [hit.meta.id for hit in response]
    result_values = [hit.meta.score for hit in response]
    return result_ids, result_values
