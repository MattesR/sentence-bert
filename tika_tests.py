import os
from tika import parser as pdfparser
from util.splitter import paragraph_splitter
from tqdm import tqdm

nsu_path = './datasets/NSU-Berichte'


def read_in_documents(path, min_size=0, min_paragraph=200, max_paragraph=400, purge_specials=True):
    document_list = []
    for dirpath, dnames, fnames in os.walk(path):
        for file in tqdm(fnames, desc=f'walking {dirpath}'):
            file_path = os.path.join(dirpath, file)
            size = os.stat(file_path).st_size
            if size >= min_size:
                parsed_pdf = pdfparser.from_file(file_path)
                document = paragraph_splitter(parsed_pdf['content'], min_paragraph=min_paragraph,
                                              max_paragraph=max_paragraph, purge_specials=purge_specials)
                document_list.append(document)
    return document_list


def get_small_files(path, min_size):
    count = 0
    max_size = 0
    max_name = ''
    small_list = []
    for dirpath, dnames, fnames in os.walk(path):
        for file in fnames:
            path = os.path.join(dirpath, file)
            size = os.stat(path).st_size
            if size > max_size:
                max_size = size
                max_name = file
            if size < min_size:
                count += 1
                small_list.append(file)
    print(
        f'''{count} files are smaller than size limit {min_size}. 
the largest file is {max_name} with {max_size} bytes. 
        ''')
    return count, small_list



