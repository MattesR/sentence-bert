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


def find_closest_sentence_end(string, index, characters=['!', '.', '?']):
    """
    finds the closest sentence ending character in a string, relative to a index position in this string.
    Sentence ending chracters are . ! and ?, if not specified. Another List can be provided.
    If none is found, the function Returns False. Else it returns the index of
    the closest sentence ending character.
    The function doesn't care for any occaision like dates or domain names.
    @param string: the string
    @param index: the index
    @param characters: the default characters
    @return: index of the closest sentence ending character or False
    """
    closest_index = False
    for character in characters:
        left_pos = False if string.rfind(character, 0, index) == -1 else abs(string.rfind(character, 0, index)-index)
        right_pos = False if string.find(character, index) == -1 else abs(string.find(character, index)-index)
        if closest_index is False:
            closest_index = min(left_pos, right_pos)
        else:
            closest_index = min(left_pos, right_pos, closest_index)
    return closest_index



