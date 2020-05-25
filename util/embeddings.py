import faiss
import numpy as np


def string_to_faiss_embedding(model, string):
    """
    Takes a string and creates an embeddings with the correct data structure for faiss to eat. The problem of it is,
    that neither faiss nor sentence-bert are designed to ever create a single embeddings from a single string.
    Sentence-Bert expects a list and returns a list
    Faiss expects an array of arrays, or a Matrix, I guess.
    @param model: the model which should create the embeddings
    @param string: the string from which the embeddings should be created
    @return: the embeddings, in a format that faiss can work with
    """
    # make a list with a single entry out of the paragraph
    unit_embedding = model.encode([string])
    # cast the embeddings to be faiss-compliant
    unit_embedding = np.array([unit_embedding[0]])
    return unit_embedding


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


class ResultHeap:
    """ Combine query results from a sliced dataset """

    def __init__(self, nq, k):
        " nq: number of query vectors, k: number of results per query "
        self.I = np.zeros((nq, k), dtype='int64')
        self.D = np.zeros((nq, k), dtype='float32')
        self.nq, self.k = nq, k
        #  changed to minheap from maxheap. The reason is that using cosine-similarity, the most similar (e.g. closest)
        #  vectors have a score of 1, whereas with distances the closest score is 0.
        heaps = faiss.float_minheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(self.D)
        heaps.ids = faiss.swig_ptr(self.I)
        heaps.heapify()
        self.heaps = heaps

    def add_batch_result(self, D, I, i0):
        assert D.shape == (self.nq, self.k)
        assert I.shape == (self.nq, self.k)
        I += i0
        self.heaps.addn_with_ids(
            self.k, faiss.swig_ptr(D),
            faiss.swig_ptr(I), self.k)

    def finalize(self):
        self.heaps.reorder()

