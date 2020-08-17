# Repository for master thesis Code: Term-based and Embedding-based Similarity Search in Large, Unstructured Data Sets

This repository provides code for creating Faiss-indices using different context-embedding models, as well as code for generating Elasticsearch indices.

Index Generation is conducted via the `Index_generator`-class. 
The Test-caller bash file is useful for generating indices automatically.

The code for comparing the models are found in `experiments.py`. There are also functions for generating test data sets. 

The used splitter can be found at util/splitter.

The setup folder contains a list of package list in order to make a fresh Ubuntu installation ready for using this repository as easily as possible.
