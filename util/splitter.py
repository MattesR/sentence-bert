def paragraph_splitter(text, min_line=0, min_paragraph=0):
    """
    splits texts heuristically into paragraphs. A paragraph is non empty lines of text followed by one or more empty
    lines of text.
    :param text: text to split
    :param min_line: The minimal length of lines that the splitter will consider. Anything below this limit will be
                    deleted
    :param min_paragraph: the minimal length of a paragraph. A shorter paragraph will be attached to the next paragraph
    :return: list of paragraphs with minimal Paragraph length min_paragraph, if provided.
    """
    paragraphs = []
    current_paragraph = ''
    for line in text.splitlines():
        line = line.strip()
        if len(line) <= min_line:
            if current_paragraph and len(current_paragraph) >= min_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = ''
        else:
            current_paragraph += line
    if current_paragraph:
        paragraphs.append(current_paragraph)
    return paragraphs


def create_pseudo_pages(textunits, n=6):
    """
    Function that creates a list of documents based on a list of TextUnits/Strings. it concatenates chunks of n strings
    into a new String and appends it to the list.
    @param textunits: list of Strings
    @param n: number of Textunits that should be concatenated to a doc
    @return: a list of new pseudo-documents.
    """
    list_of_pseudo_docs = []
    for i in range(0, len(textunits), n):
        list_of_pseudo_docs.append(textunits[i:i+n])
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

