from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings

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





