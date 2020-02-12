from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings
import faiss
import numpy as np

result_path = './results'
faiss_path = './faiss_indexes'
INDEX_SIZE = 200000

# init embedding
embedding = BertEmbeddings(layers='-1')
# fine tune mode 'nonlinear' for now, as docs state to use it, if you have "simple word embeddings
# that are not task-trained
document_embeddings = DocumentPoolEmbeddings([embedding], fine_tune_mode='nonlinear')


def embed_all(embedding, sentences):
    failed_list = []
    for index, sentence in enumerate(sentences):
        try:
            embedding.embed(sentence)
            print(f'successfully embedded sentence {index}')
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


def array_from_list(list_of_arrays):
    """
    gives an array from a list of arrays. they must all have the same length.
    :param list_of_arrays: the list of arrays .....
    :return:
    """
    shape = list(list_of_arrays[0].shape)
    shape[:0] = [len(list_of_arrays)]
    arr = np.concatenate(list_of_arrays).reshape(shape)
    return arr
