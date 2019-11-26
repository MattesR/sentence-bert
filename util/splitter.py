



def paragraph_splitter(text):
    """
    splits texts heuristically into paragraphs. A paragraph is non empty lines of text followed by one or more empty
    lines of text.
    :param text:
    :return:
    """
    paragraphs = []
    current_paragraph = ''
    for line in text.splitlines():
        if not line:
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = ''
        else:
            current_paragraph += line
    if current_paragraph:
        paragraphs.append(current_paragraph)
    return paragraphs
