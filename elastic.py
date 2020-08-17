from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from elasticsearch_dsl import Search, Index, Document, connections, \
    Keyword, Date, Text, Integer, Nested, InnerDoc, Boolean


connections.create_connection(hosts=['localhost:9200'])
enron_index = Index('enron')
NSU_index = Index('nsu')
experiment_index = Index('experiment')


def create_index(name):
    """
    creates an index and puts it into elastic search, with documentclass Hooverdoc.
    :param name: name of the index you want to create.
    :return:
    """
    index = Index(name)
    index.document(HooverDoc)
    try:
        index.create()
    except RequestError as e:
        print(e.status_code)
        if e.error == 'resource_already_exists_exception':
            print(f'the index {name} already exists')
        else:
            print(f'something went wrong:')
            print(e.error)
        return
    if index.exists():
        print(f'index {name} was created')


class TextUnit(InnerDoc):
    content = Text(term_vector="yes")
    position = Integer()


@NSU_index.document
class Report(Document):
    """
    Report document for the enron index. It stores the following data about the Report:
    author
    receiver (mail address)
    date
    subject
    payload
    """
    author = Keyword()
    creation_date = Date()
    title = Text()
    body = Nested(TextUnit)

    def add_unit(self, content, position):
        self.body.append(
          TextUnit(content=content, position=position))


@enron_index.document
class Mail(Document):
    """
    Mail document for the enron index. It stores the following data about the mail:
    sender (mail address)
    receiver (mail address)
    date
    subject
    payload
    """
    sender = Keyword()
    receiver = Keyword()
    sent_date = Date()
    subject = Text()
    body = Nested(TextUnit)

    def add_unit(self, content, position):
        self.body.append(
          TextUnit(content=content, position=position))


class HooverDoc(Document):
    """
    Document with the same fields as documents in hoover-search. This document will be used for created indices.
    The only difference is, that the Text field is replaced with the nested TextUnits.
    """
    attachments = Boolean()
    content_type = Keyword()
    date = Date()
    date_created = Date()
    email_domains = Keyword()
    filetype = Keyword()
    id = Keyword()
    in_reply_to = Keyword()
    lang = Keyword()
    md5 = Keyword()
    message = Keyword()
    message_id = Keyword()
    path = Keyword()
    path_text = Text()
    path_parts = Keyword()
    references = Keyword()
    rev = Integer()
    sha1 = Keyword()
    size = Integer()
    suffix = Keyword()
    thread_index = Keyword()
    word_count = Integer()
    body = Nested(TextUnit)

    def add_unit(self, content, position):
        self.body.append(
          TextUnit(content=content, position=position))


class TestCase(Document):
    content = Text(term_vector="yes", analyzer='stop')


