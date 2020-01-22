from tika import parser as pdfparser
from util.splitter import paragraph_splitter
from somajo import SoMaJo

tokenizer = SoMaJo("de_CMC")
nsu_path = '../datasets/NSU-Berichte'
parsed_pdf = pdfparser.from_file(nsu_path +'/NRW_MMD16-14400.pdf')

bericht = paragraph_splitter(parsed_pdf['content'], min_line=30, min_paragraph=200)

author = parsed_pdf['metadata']['Author']
title = parsed_pdf['metadata']['title']
date = parsed_pdf['metadata']['Creation-Date']

sentences = tokenizer.tokenize_text(bericht)
newSentences = []
for sentence in sentences:
    newSentence = ''
    for token in sentence:
        newSentence += ' ' + token.text
    newSentences.append(newSentence)
