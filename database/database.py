import json
import psycopg2
from sentence_transformers import SentenceTransformer

DBNAME='enron'
DBUSER='postgres'
DBPASSWORD='postgres'


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


def put_embeddings():
    """
    most likely gargabe without any use :(
    :return:
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    try:
        conn = psycopg2.connect(host='localhost', database='enron', user='postgres', password='postgres')
        cur = conn.cursor()
        cur.execute("SELECT sentence_text FROM sentences")
        sentence = cur.fetchone()[0]
        while row:
            sentence_embedding = model.encode([sentence])
            json_embedding = json.dumps(sentence_embedding[0].tolist())
            row = cur.fetchone()
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error_msg:
        print(error_msg)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    create_tables()
