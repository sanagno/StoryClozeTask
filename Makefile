create_data:
	cd data/glove-embeddings && \
	wget https://nlp.stanford.edu/data/wordvecs/glove.6B.zip && \
	unzip glove.6B.zip && \
	rm glove.6B.zip glove.6B.200d.txt glove.6B.300d.txt glove.6B.50d.txt
	cd data/skip-thought && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/NQ9OT8Xxvdxn3wo/download -o skip-thoughts-embeddings_train.npy && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/10CivpGpg8O1Bfe/download -o skip-thoughts-embeddings_validation.npy && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/PKQm7YuCMsPhBv6/download -o skip-thoughts-embeddings_test.npy

.PHONY: data 
