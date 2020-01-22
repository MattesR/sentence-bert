from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from elasticsearch_dsl import Search, Index, Document, connections, \
    Keyword, Date, Text, Integer, MetaField, Nested, InnerDoc
from elasticsearch_dsl.query import Nested
from elasticsearch_dsl.query import Match, Nested, Term, MoreLikeThis
from tabulate import tabulate

client = Elasticsearch()


mlt_match = MoreLikeThis(fields=["body.content"],
                         like=["you owe me"],
                         min_term_freq=1,
                         min_doc_freq=1)
innerMatch = Match(body__content='stock')
nestedMatch = Nested(path='body', query=innerMatch)



# retrieve all documents containing stock in its body)
s = Search(using=client, index='enron') \
        .query("match", body="stock")




"""
in order to change the size of return:
s= s[0:0] will create a size 0 request.
It's all done with python slicing.
"""
