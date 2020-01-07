

def paragraph_splitter(text, min_line=0, min_paragraph=None):
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
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = ''
        else:
            current_paragraph += line
    if current_paragraph:
        paragraphs.append(current_paragraph)
    if min_paragraph:
        finished = False
        while not finished:
            finished = True
            for index, paragraph in enumerate(paragraphs):
                if len(paragraph) < min_paragraph:
                    if index == len(paragraphs)-1:
                        paragraphs[index-1:index+1] = [' '.join(paragraphs[index-1:index+1])]
                    else:
                        paragraphs[index:index + 2] = [' '.join(paragraphs[index:index + 2])]
                    finished = False
    return paragraphs
