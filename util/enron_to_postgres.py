import psycopg2
from psycopg2 import sql
from .splitter import paragraph_splitter


DBNAME='enron'
DBUSER='postgres'
DBPASSWORD='postgres'


def insert_document(text, name):
    paragraphs = paragraph_splitter(text)
    document_command = """
    INSERT INTO documents(document_name)
    VALUES
        (%s)
    RETURNING document_id;
    """
    paragraph_command = """
        INSERT INTO sentences(sentence_text, document_id)
        VALUES
            (%s, %s);
        """
    try:
        conn = psycopg2.connect(host='localhost', database=DBNAME, user=DBUSER, password=DBPASSWORD)
        cur = conn.cursor()
        cur.execute(document_command, (name,))
        # print('document stored in db')
        document_id = cur.fetchone()[0]
        for paragraph in paragraphs:
            cur.execute(paragraph_command, (paragraph, document_id,))
        # print('all sentences as well')
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            # print('Database connection closed.')

