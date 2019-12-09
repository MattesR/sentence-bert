from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from elasticsearch_dsl import Search, Index, Document, connections, \
    Keyword, Date, Text, Integer, MetaField, Nested, InnerDoc


connections.create_connection(hosts=['localhost:9200'])
enron_index = Index('enron')


def create_index(name):
    """
    creates an index and puts it into elastic search
    :param name: name of the index you want to create.
    :return:
    """
    index = Index(name)
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
    content = Text(analyzer='snowball')
    position = Integer()


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


