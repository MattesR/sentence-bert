from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings
import faiss
import numpy as np



result_path = './results'
faiss_path = './faiss_indexes'
INDEX_SIZE = 200000

# init embeddings
embedding = BertEmbeddings(layers='-1')
# mode none, since I do not intend to task train my embeddings
document_embeddings = DocumentPoolEmbeddings([embedding], fine_tune_mode='none')


def embed_all(embedding, sentences):
    failed_list = []
    for index, sentence in enumerate(sentences):
        try:
            embedding.embed(sentence)
        except RuntimeError:
            print(f'could not embed sentence with index {index}\nstoring in failed index list')
            failed_list.append(index)
    if failed_list:
        return failed_list
    else:
        return True


def create_faiss_index(embeddings, name, path=faiss_path, index_size=INDEX_SIZE):
    index = faiss.IndexFlatL2(768)
    if len(embeddings) <= index_size:
        index.add(embeddings)
        faiss.write_index(index, path + f'/{name}')
        print(f'saved one faiss index with name {name}')
    else:
        index_number = 0
        for i in range(0, len(embeddings), index_size):
            index.embed(embeddings[i:i + index_size])
            faiss.write_index(index, path + f'/{name}_{index_number}')
            index = faiss.IndexFlatL2(768)
            i += 1






# if this script is called directly from the command line, it will be because generating faiss index failed.
# the script will automatically recover from a failure by calling itself again and starting where it had to leave off
# due to failure.
