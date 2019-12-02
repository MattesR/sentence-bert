import os
import time
from tqdm import tqdm
from elastic import Mail
from elasticsearch_dsl import connections
from util.splitter import paragraph_splitter
from util.parser import parsemail
ENRON_MAIL_PATH = "../datasets/enron"
test_path = "/home/moiddes/opt/datasets/enron/white-s/val"
results_path= "./results"




"""
This function walks the enron dataset, storing all mails and metadata in elastic
"""
if __name__ == "__main__":
    t_start = time.time()
    connections.create_connection(hosts=['localhost:9200'])
    Mail.init()
    el_id = 0
    falsely_parsed = 0
    for dirpath, dnames, fnames in os.walk(ENRON_MAIL_PATH):
        for file in tqdm(fnames, desc=f'walking {dirpath}'):
            try:
                parsed_mail = parsemail(os.path.join(dirpath, file))
            except UnicodeDecodeError:
                print(f"mail {os.path.join(dirpath, file)} wasn't parsed correctly")
                falsely_parsed += 1
            if parsed_mail['to']:
                receiver = parsed_mail['to'].split(',')
            elastic_mail = Mail(sent_date=parsed_mail['date'],
                                sender=parsed_mail['from'],
                                receiver=parsed_mail['to'],
                                subject=parsed_mail['subject'],
                                meta={'id': el_id}
                                )
            for position, paragraph in enumerate(paragraph_splitter(parsed_mail.get_payload()), start=1):
                elastic_mail.add_unit(paragraph, position)
            reason = elastic_mail.save()
            if not reason == 'created':
                print(f'document {file} was not created. It was {reason}')
            el_id += 1
    t_stop = time.time()
    total_time = t_stop - t_start
    with open(results_path + '/elastic_walk.txt', 'w') as output:
        output.write(f"""
        time needed to index the Enron Dataset into Elasticsearch using paragraph splitter: {total_time} seconds 
        """)
    print(f'all emails were processed. {falsely_parsed} mails were falsely parsed and are missing from the corpus. \n'
          f'it took {total_time} seconds to index the dataset.')


