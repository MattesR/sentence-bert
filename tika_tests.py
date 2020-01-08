from tika import parser as pdfparser
from util.splitter import paragraph_splitter


nsu_path = '../datasets/NSU-Berichte'
parsed_pdf = pdfparser.from_file(nsu_path +'/NRW_MMD16-14400.pdf')

bericht = paragraph_splitter(myFirstPDF['content'], min_line=30, min_paragraph=200)

author = parsed_pdf['metadata']['Author']
title = parsed_pdf['metadata']['title']
date = parsed_pdf['metadata']['Creation-Date']
