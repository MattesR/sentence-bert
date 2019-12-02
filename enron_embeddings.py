
import os
import time
from datetime import datetime

import psycopg2
from tqdm import tqdm
from util.parser import parsemail
from util.splitter import paragraph_splitter
import util.enron_to_postgres as db
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
result_path = './results'
faiss_path = './faiss_indexes'
model_name = 'bert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)
DBNAME='enron'
DBUSER='postgres'
DBPASSWORD='postgres'
ENRON_MAIL_PATH = "../datasets/enron"
test_path = "/home/moiddes/opt/datasets/enron/white-s/val"


if __name__ == "__main__":
    """
    This Script is reponsible for walking the enron data base path, creating database entries for all documents and
    calculating and storing embeddings.
    """
    t_start = time.time()
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
                # make a list with a single entry out of the paragraph
                unit_embedding = model.encode([paragraph])
                # cast the embedding to be faiss-compliant
                # TODO: Make this less messy like for real that's horrible
                unit_embedding = np.array([unit_embedding[0]])
                id_index.add_with_ids(unit_embedding, np.array([unit_id]))
    now = datetime.now()
    faiss.write_index(id_index, faiss_path + '/faiss_index_' + model_name + '_' + now.strftime("%d,%m,%Y %Huhr%M"))
    t_stop = time.time()
    total_time = t_stop - t_start
    with open(result_path + '/faiss_walk.txt', 'w') as output:
        output.write(f""" 
        time needed to index the Enron Dataset  into Postgres and using paragraph splitter: {total_time} seconds 
        """)
    print(
        f'all emails were processed. {falsely_parsed} mails were falsely parsed and are missing from the corpus. \n'
        f'it took {total_time} seconds to index the dataset.')
