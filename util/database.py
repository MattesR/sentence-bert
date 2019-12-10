import json
import psycopg2
from sentence_transformers import SentenceTransformer
from contextlib import contextmanager

DBNAME='enron'
DBUSER='postgres'
DBPASSWORD='postgres'


import psycopg2


@contextmanager
def connect_to_db(host='localhost', database=DBNAME, user=DBUSER, password=DBPASSWORD):
    try:
        conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    yield conn
    conn.close()

def insert_document(connection, name):
    """
    Takes a connection and a name and inserts the document into the database.
    returns the database_id of the inserted document
    :param connection: database connection
    :param name: name
    :return: document_id
    """
    document_command = """
    INSERT INTO documents(document_name)
    VALUES
        (%s)
    RETURNING document_id;
    """
    try:
        cur = connection.cursor()
        cur.execute(document_command, (name,))
        document_id = cur.fetchone()[0]
        cur.close()
        connection.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return document_id


def insert_text_unit(connection, content, position, document_id):
    """
    Inserts a text unit into the database. Needs a connection and the content plus position in the document it came from
    returns it unit_id from the database
    :param connection: database connection
    :param content: text content
    :param position: position inside the document
    :param document_id: document_id of the document
    :return: unit_id from the database
    """
    unit_command = """
        INSERT INTO text_units(content, position, document_id)
        VALUES
            (%s, %s, %s)
        RETURNING unit_id;
        """
    try:
        cur = connection.cursor()
        cur.execute(unit_command, (content, position, document_id))
        unit_id = cur.fetchone()[0]
        cur.close()
        connection.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return unit_id


def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE documents (
            document_id SERIAL PRIMARY KEY,
            document_name TEXT
        )
        """,
        """ CREATE TABLE text_units (
                unit_id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                position INTEGER,
                document_id INTEGER,
                FOREIGN KEY (document_id)
                REFERENCES documents (document_id)
                ON UPDATE CASCADE ON DELETE CASCADE
                )
        """)
    conn = None
    try:
        conn = psycopg2.connect(host='localhost', database='enron', user='postgres', password='postgres')
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error_msg:
        print(error_msg)
    finally:
        if conn is not None:
            conn.close()


def get_sentences_from_ids(connection, ids):
    """

    @param connection: connection to use
    @param ids: tuple of ids
    @return: weiß noch nicht :D
    """
    document_command = """
        select * from text_units where unit_id in %s;
    """
    try:
        cur = connection.cursor()
        cur.execute(document_command, (ids,))
        result_list = cur.fetchall()
        cur.close()
        connection.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return result_list


if __name__ == "__main__":
    create_tables()
