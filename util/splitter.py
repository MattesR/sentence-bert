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
            line = re.sub(r'([!.,])([A-Za-z]{3})', r'\1 \2', line) # Adds Whitespace between words if it's missing
        if len(line) <= min_line:
            if current_paragraph and len(current_paragraph) >= min_paragraph:
                end_of_paragraph = find_closest_sentence_end(current_paragraph, len(current_paragraph) - 1)
                # if the end of the paragraph would be a paragraph of proper length, add that instead of the whole
                # paragraph. Since end_of_paragraph is False if no sentence_end was found, it will not be greater
                # than min_paragraph. That's why it's 'min_paragraph-1' insteadof '>='
                # @NOTE: Wouldn't work with min_paragraph = 0 !!
                if end_of_paragraph > min_paragraph-1:
                    end_of_paragraph += 1
                    new_paragraph = current_paragraph[end_of_paragraph + 1:]
                    current_paragraph = current_paragraph[:end_of_paragraph]
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = dehyphen(new_paragraph)
                else:  # if it is not possible to find a sentence end and create a min paragraph add the whole paragraph
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ''
        else:
            if max_paragraph is not False:  # if a maximum paragraph length was defined
                if len(current_paragraph) + len(line) < max_paragraph:  # check, whether you could add the whole line
                    current_paragraph += dehyphen(line)  # add it, if you stay below max_paragraph
                else:  # if not, check how much the next line is over the max_paragraph length
                    overdraw = len(current_paragraph) + len(line) - max_paragraph
                    # find the closest sentence end to the overdraw in the line that you want to add, if its reasonably
                    # distanced from the overdraw.
                    end_of_paragraph = find_closest_sentence_end(line, len(line)-overdraw-1, max_distance=20)
                    # if you find a sentence end in the line, add the line up to the sentence end to the current
                    # paragraph, add the current paragraph to the list and have the rest of the line be the new
                    # current paragraph.
                    # the if below means, that if the index in the new line plus the already current paragraph makes
                    # up a complete paragraph, add it
                    if end_of_paragraph and end_of_paragraph + len(current_paragraph) > min_paragraph:
                        end_of_paragraph += 1  # so that the sentence ending symbol is part of the paragraph
                        current_paragraph += line[:end_of_paragraph]  # add it to the current paragraph
                        paragraphs.append(current_paragraph.strip())  # strip so that leading whitespace is deleted
                        current_paragraph = dehyphen(line[end_of_paragraph+1:])  # rest of line is new current paragraph
                        #  If the rest of the line is really long, you have to make sure that it is also split into
                        #  paragraphs within min and max paragraph length. The nicest is to split these long lines
                        #  into sentences with sentence endings, however that cannot be guaranteed for all inputs.
                        #  therefore it might be necessary to cut at whitespace or just wherever.
                        while len(current_paragraph) > max_paragraph:
                            rest_of_line = current_paragraph
                            overdraw = len(current_paragraph) - max_paragraph
                            end_of_paragraph = find_closest_sentence_end(rest_of_line,
                                                                         len(rest_of_line) - overdraw - 1,
                                                                         max_distance=20)
                            if end_of_paragraph > min_paragraph-1:  # so that no tiny paragraphs are added
                                end_of_paragraph += 1
                                current_paragraph = rest_of_line[:end_of_paragraph]
                                paragraphs.append(current_paragraph.strip())
                                current_paragraph = dehyphen(rest_of_line[end_of_paragraph + 1:])
                            else:
                                #  this means that either there was no sentence end in the rest of the line or the
                                #  resulting paragraph was really short. If that is the case, create a paragraph of
                                #  maximum length, which ends at the nearest whitespace and itereate through the rest.
                                #  the max distance is only relevant for seriously malformed inputs
                                near_max_whitespace = find_closest_sentence_end(rest_of_line,
                                                                                max_paragraph,
                                                                                characters=[' '], max_distance=10)
                                if near_max_whitespace:
                                    near_max_whitespace += 1
                                    current_paragraph = rest_of_line[:near_max_whitespace]
                                    paragraphs.append(current_paragraph.strip())
                                    current_paragraph = dehyphen(rest_of_line[near_max_whitespace + 1:])
                                else:
                                    current_paragraph = rest_of_line[:max_paragraph]
                                    paragraphs.append(current_paragraph.strip())
                                    current_paragraph = dehyphen(rest_of_line[max_paragraph + 1:])
                    else:  # if there's no sentence ending symbol in the next line, find it at the end of the paragraph
                        end_of_paragraph = find_closest_sentence_end(current_paragraph,
                                                                     len(current_paragraph)-1,
                                                                     max_distance=20)
                        if end_of_paragraph > min_paragraph-1:
                            end_of_paragraph += 1
                            new_paragraph = dehyphen(current_paragraph[end_of_paragraph+1:])
                            current_paragraph = current_paragraph[:end_of_paragraph]
                            paragraphs.append(current_paragraph.strip())
                            current_paragraph = new_paragraph
                        else:  # if there's none in the paragraph either, hard-cut the paragraph at the maximum length
                            current_paragraph += line[:-overdraw]
                            paragraphs.append(current_paragraph.strip())
                            current_paragraph = line[-overdraw:]
                            #  here, the while loop is a little bit simpler, due to the fact that we already checked
                            #  the old current_paragraph __and__ the new line for sentence ending symbols and did not
                            #  find any. This is the bottom of the barrel split at whitespace or just chunk'em
                            while len(current_paragraph) > max_paragraph:
                                rest_of_line = current_paragraph
                                near_max_whitespace = find_closest_sentence_end(rest_of_line,
                                                                                max_paragraph,
                                                                                characters=[' '], max_distance=20)
                                if near_max_whitespace:
                                    near_max_whitespace += 1
                                    current_paragraph = rest_of_line[:near_max_whitespace]
                                    paragraphs.append(current_paragraph.strip())
                                    current_paragraph = dehyphen(rest_of_line[near_max_whitespace + 1:])
                                else:
                                    current_paragraph = rest_of_line[:max_paragraph]
                                    paragraphs.append(current_paragraph.strip())
                                    current_paragraph = dehyphen(rest_of_line[max_paragraph + 1:])
            else:
                current_paragraph += dehyphen(line)

    if current_paragraph:
        if len(current_paragraph) < min_paragraph:
            if paragraphs:
                paragraphs[-1] += current_paragraph
            else:
                paragraphs.append(current_paragraph.strip())
        else:
            paragraphs.append(current_paragraph.strip())
    return paragraphs


def create_pseudo_pages(textUnits, n=4):
    """
    Function that creates a list of documents based on a list of TextUnits/Strings. it concatenates chunks of n strings
    into a new String and appends it to the list.
    @param textUnits: list of Strings
    @param n: number of Textunits that should be concatenated to a doc
    @return: a list of new pseudo-documents.
    """
    rest = (len(textUnits) % n)
    list_of_pseudo_docs = []
    if len(textUnits) < n:
        list_of_pseudo_docs.append(textUnits)
        return list_of_pseudo_docs
    for i in range(0, len(textUnits) - rest, n):  # add rest to last page instead of creating new page
        list_of_pseudo_docs.append(textUnits[i:i + n])
    if rest:
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
        if len(document) >= 4:
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


def find_closest_sentence_end(line, index, characters=['!', '.', '?'], max_distance=False):
    """
    finds the closest sentence ending character in a string, relative to a index position in this string.
    Sentence ending chracters are . ! and ?, if not specified. Another List can be provided.
    If none is found, the function Returns False. Else it returns the index of
    the closest sentence ending character.
    The function doesn't care for any occasion like dates or domain names.
    @param line: the string
    @param index: the index
    @param characters: the default characters
    @param max_distance: the maximum distance that a character might be away from the index.
    @return: index of the closest sentence ending character or False
    """
    # Should 'basically never happen' but this makes the function a little more robust.
    if index > len(line):
        index = len(line)-1
    closest_off = len(line)
    if max_distance is False:  # if there was no max distance defined (so it's false), set it to closest_off.
        max_distance = closest_off
    closest_position = index
    matching_group = r'\w{2,}['
    for character in characters:
        matching_group += character
    matching_group += r']'
    all_matches = re.finditer(matching_group, line)
    for match in all_matches:
        end_position = match.span()[1]
        end_off = abs(index - end_position)
        if end_off < max_distance:
            if end_off < closest_off:
                closest_off = end_off
                closest_position = end_position
    if closest_off == len(line):
        return False
    return closest_position - 1


def dehyphen(string):
    """
    strips hyphen at end of string and adds space if it doesn't find a hyphen. If the string is empty, it just returns
    empty.
    @param string: the string
    @return: the new string
    """
    if string:
        if string[-1] == '-':
            return string[:-1]
        else:
            return string + ' '
    else:
        return string
