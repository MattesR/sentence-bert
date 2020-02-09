from flair.data import Sentence
import re


def paragraph_splitter(text, min_line=0, min_paragraph=0, max_paragraph=False, purge_specials=False):
    """
    splits texts heuristically into paragraphs. A paragraph is non empty lines of text followed by one or more empty
    lines of text.
    :param text: text to split
    :param min_line: The minimal length of lines that the splitter will consider. Anything below this limit will be
                    deleted
    :param min_paragraph: the minimal length of a paragraph. A shorter paragraph will be attached to the next paragraph
    :param max_paragraph: the maximal length of a a paragraph. a paragraph which is too long will be cut at the last or
    next sentence.
    :return: list of paragraphs with minimal Paragraph length min_paragraph, if provided.
    """
    paragraphs = []
    current_paragraph = ''
    for line in text.splitlines():
        line = line.strip()
        if purge_specials:
            line = re.sub(r'(\W)(?=\1)', '', line)
            line = re.sub(r'([!.,])([A-Za-z]{3})', r'\1 \2', line)
        if len(line) <= min_line:
            if current_paragraph and len(current_paragraph) >= min_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = ''
        else:
            if max_paragraph is not False:
                if len(current_paragraph) + len(line) < max_paragraph:
                    if line[len(line)-1] == '-':  #if the last character in line is a hyphen, delete it and add line,
                        current_paragraph += line[:len(line)-1]
                    else:
                        current_paragraph += line + ' '  # it it doesn't, add it with a space between the ending characters
                else:
                    overdraw = len(current_paragraph) + len(line) - max_paragraph
                    end_of_paragraph = find_closest_sentence_end(line, len(line)-overdraw-1) + 1
                    if end_of_paragraph:
                        current_paragraph += line[:end_of_paragraph]
                        paragraphs.append(current_paragraph)
                        current_paragraph = line[end_of_paragraph+1:]
                    else:
                        end_of_paragraph = find_closest_sentence_end(current_paragraph, len(current_paragraph)-1) + 1
                        new_paragraph = current_paragraph[end_of_paragraph+2:]
                        current_paragraph = current_paragraph[:end_of_paragraph]
                        paragraphs.append(current_paragraph)
                        current_paragraph = new_paragraph
            else:
                if line[len(line)-1] == '-':  #if the last character in line is a hyphen, delete it and add line,
                    current_paragraph += line[:len(line)-1]
                else:
                    current_paragraph += line + ' '

    if current_paragraph:
        paragraphs.append(current_paragraph)
    return paragraphs


def create_pseudo_pages(textUnits, n=6):
    """
    Function that creates a list of documents based on a list of TextUnits/Strings. it concatenates chunks of n strings
    into a new String and appends it to the list.
    @param textUnits: list of Strings
    @param n: number of Textunits that should be concatenated to a doc
    @return: a list of new pseudo-documents.
    """
    rest = (len(textUnits) % n)
    list_of_pseudo_docs = []
    for i in range(0, len(textUnits) - rest, n):  # add rest to last page instead of creating new page
        list_of_pseudo_docs.append(textUnits[i:i + n])
    for restString in textUnits[-rest:]:
        list_of_pseudo_docs[-1].append(restString)  # add rest to last page instead of creating new page
    return list_of_pseudo_docs


def create_document_pairs(pseudo_documents):
    """
    Create documents pairs from a list of pseudo documents consisting of a list of text units or paragraphs. 
    It will split each document in even and odd TextUnits and append them to the list.
    From that, the following structure results:
    pseudo_document[n] -> split into: pair_list[2*n], pair_list[2*n+1]
    @param pseudo_documents: The list of pseudo-documents for which the list of document pairs will be constructed
    @return: list of document pairs 
    """
    pair_list = []
    for document in pseudo_documents:
        even, odd = ' '.join(document[::2]),  ' '.join(document[1::2])
        pair_list.append(even)
        pair_list.append(odd)
    return pair_list


def create_flair_sentences(sentences):
    """
    create a list of flairs Sentence objects from a list of strings.

    @param sentences: list of strings
    @return: list of sentences
    """
    flair_sentences = []
    for sentence in sentences:
        flair_sentences.append(Sentence(sentence))
    return flair_sentences


def flair_pairs_from_strings(textUnits, n):
    return create_flair_sentences(create_document_pairs(create_pseudo_pages(textUnits, n)))


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
    closest_off = len(string)
    closest_pos = index
    for character in characters:
        left_off = closest_off if string.rfind(character, 0, index) == -1 else abs(string.rfind(character, 0, index)-index)
        left_pos = index - left_off
        right_off = closest_off if string.find(character, index) == -1 else abs(string.find(character, index)-index)
        right_pos = index + right_off
        if left_off < right_off:
            if left_off < closest_off:
                closest_off = left_off
                closest_pos = left_pos
        else:
            if right_off < closest_off:
                closest_off = left_off
                closest_pos = left_pos
    if closest_off == len(string):
        closest_off = False
        closest_pos = False
    return closest_pos



