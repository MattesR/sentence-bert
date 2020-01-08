import os
import time
from tqdm import tqdm
from elastic import Report
from elasticsearch_dsl import connections
from util.splitter import paragraph_splitter
from tika import parser as pdfparser
ENRON_MAIL_PATH = "../datasets/NSU-Berichte"
results_path= "./results"




"""
This function walks the NSU dataset, storing all Reports and some metadata in elastic.
The Document contains nested inner documents which are the paragraphs of the Report, split by a paragraph splitter
"""
if __name__ == "__main__":
    t_start = time.time()
    connections.create_connection(hosts=['localhost:9200'])
    Report.init()
    el_id = 0
    falsely_parsed = 0
    for dirpath, dnames, fnames in os.walk(ENRON_MAIL_PATH):
        for file in tqdm(fnames, desc=f'walking {dirpath}'):
            try:
                parsed_pdf = pdfparser.from_file(os.path.join(dirpath, file))
            except UnicodeDecodeError as e:
                print(f'there was an an error, apparently {e}')
                falsely_parsed += 1
            elastic_report = Report(author=parsed_pdf['metadata'].get('Author', f'unknown Author {el_id}'),
                                    title=parsed_pdf['metadata'].get('title', f'unknown title {el_id}'),
                                    date=parsed_pdf['metadata'].get('Creation-Date', f'unknown creation date {el_id}'),
                                    meta={'id': el_id}
                                    )
            for position, paragraph in enumerate(paragraph_splitter(parsed_pdf['content'],
                                                                    min_line=30, min_paragraph=200)
                                                 , start=1):
                elastic_report.add_unit(paragraph, position)
            reason = elastic_report.save()
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


