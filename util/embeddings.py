import faiss
import numpy as np


def string_to_faiss_embedding(model, string):
    """
    Takes a string and creates an embedding with the correct data structure for faiss to eat. The problem of it is,
    that neither faiss nor sentence-bert are designed to ever create a single embedding from a single string.
    Sentence-Bert expects a list and returns a list
    Faiss expects an array of arrays, or a Matrix, I guess.
    @param model: the model which should create the embedding
    @param string: the string from which the embedding should be created
    @return: the embedding, in a format that faiss can work with
    """
    # make a list with a single entry out of the paragraph
    unit_embedding = model.encode([string])
    # cast the embedding to be faiss-compliant
    unit_embedding = np.array([unit_embedding[0]])
    return unit_embedding


class ResultHeap:
    """ Combine query results from a sliced dataset """

    def __init__(self, nq, k):
        " nq: number of query vectors, k: number of results per query "
        self.I = np.zeros((nq, k), dtype='int64')
        self.D = np.zeros((nq, k), dtype='float32')
        self.nq, self.k = nq, k
        heaps = faiss.float_maxheap_array_t()
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


