
import os
import time
from datetime import datetime

import psycopg2
from tqdm import tqdm
from util.embeddings import string_to_faiss_embedding
from util.parser import parsemail
from util.splitter import paragraph_splitter
import util.database as db
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
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


if __name__ == "__main__":
    """
    This Script is reponsible for walking the enron data base path, creating database entries for all documents and
    calculating and storing embeddings.
    """
    t_start = time.time()
    current_index = 0
    current_index_size = 0
    try:
        conn = psycopg2.connect(host='localhost', database=DBNAME, user=DBUSER, password=DBPASSWORD)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    index = faiss.IndexFlatL2(768)
    id_index = faiss.IndexIDMap(index)
    falsely_parsed = 0
    for dirpath, dnames, fnames in os.walk(ENRON_MAIL_PATH):
        for file in tqdm(fnames, desc=f'walking {dirpath}'):
            try:
                parsed_mail = parsemail(os.path.join(dirpath, file))
            except UnicodeDecodeError:
                print(f"mail {os.path.join(dirpath, file)} wasn't parsed correctly")
                falsely_parsed += 1

            document_id = db.insert_document(conn, parsed_mail['subject'])

            for position, paragraph in enumerate(paragraph_splitter(parsed_mail.get_payload()), start=1):
                unit_id = db.insert_text_unit(conn, paragraph, position, document_id)
                unit_embedding = string_to_faiss_embedding(model, paragraph)
                id_index.add_with_ids(unit_embedding, np.array([unit_id]))
                current_index_size += 1
                if current_index_size >= index_size:
                    now = datetime.now()
                    faiss.write_index(id_index, faiss_path + '/faiss_index_' + model_name + '_' + str(
                        current_index) + '_' + now.strftime("%d,%m,%Y %Huhr%M"))
                    print(f'written index File number {current_index} it contains {current_index_size} vectors')
                    index = 0
                    index = faiss.IndexFlatL2(768)
                    id_index = 0
                    id_index = faiss.IndexIDMap(index)
                    current_index += 1
                    current_index_size = 0
    now = datetime.now()
    faiss.write_index(id_index, faiss_path + '/faiss_index_' + model_name + '_' + str(
        current_index) + '_' + now.strftime("%d,%m,%Y %Huhr%M"))
    print(f'written final index File number {current_index} it contains {current_index_size} vectors')
    t_stop = time.time()
    total_time = t_stop - t_start
    with open(result_path + '/faiss_walk.txt', 'w') as output:
        output.write(f""" 
        time needed to index the Enron Dataset  into Postgres and using paragraph splitter: {total_time} seconds 
        """)
    print(
        f'all emails were processed. {falsely_parsed} mails were falsely parsed and are missing from the corpus. \n'
        f'it took {total_time} seconds to index the dataset.')
