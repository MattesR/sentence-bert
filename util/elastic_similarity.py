from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Nested
from elasticsearch_dsl.query import Nested, MoreLikeThis

import time
from tabulate import tabulate

client = Elasticsearch()
tabulate_headers = ['rank', 'content', 'position', 'document_id', 'mlt_score']


def create_nested_search(query, index):
    s = Search(using=client, index=index)
    s.source(includes=['*'], excludes=["body"])
    mlt_match = MoreLikeThis(fields=["body.content"],
                             like=[query],
                             min_term_freq=1,
                             min_doc_freq=1)
    nested_query = Nested(path='body', inner_hits={}, query=mlt_match)
    s = s.query(nested_query)
    return s


def create_mlt_with_id(document_id,position , index):
    s = Search(using=client, index=index)
    s.source(includes=['*'], excludes=["body"])
    mlt_match = MoreLikeThis(fields=["body.content"],
                             like=[id],
                             min_term_freq=1,
                             min_doc_freq=1)
    nested_query = Nested(path='body', inner_hits={}, query=mlt_match)
    s = s.query(nested_query)
    return s


def get_document_via_id(id, index):
    s = Search(using=client, index=index)
    return s


def similarity_search_text(query, index):
    t_start = time.time()
    search = create_nested_search(query, index)
    response = search.execute()
    response = response.to_dict()
    results = []
    for rank, hit in enumerate(response['hits']['hits']):
        hit_list = [rank]
        # The check is necessary right now, since response.to_dict wasn't working on these inner structs in inspector
        if not type(hit['inner_hits']['body']) is dict:
            hit['inner_hits']['body'] = hit['inner_hits']['body'].to_dict()
        hit_list.append(hit['inner_hits']['body']['hits']['hits'][0]['_source']['content'])  # add content
        hit_list.append(hit['inner_hits']['body']['hits']['hits'][0]['_source']['position'])  # add position
        hit_list.append(hit['inner_hits']['body']['hits']['hits'][0]['_id'])  # add document ID
        hit_list.append(hit['inner_hits']['body']['hits']['hits'][0]['_score'])
        results.append(hit_list)
    t_stop = time.time()
    print(tabulate(results, headers=tabulate_headers))
    print(f' Getting these results took {t_stop - t_start} seconds')


