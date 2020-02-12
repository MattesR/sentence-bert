from tika import parser as pdfparser
from util.splitter import paragraph_splitter

nsu_path = './datasets/NSU-Berichte'
parsed_pdf = pdfparser.from_file(nsu_path +'/NRW_MMD16-14400.pdf')

bericht = paragraph_splitter(parsed_pdf['content'], min_paragraph=200, max_paragraph=400,purge_specials=True)



