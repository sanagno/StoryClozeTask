# downloads glove 100d embeddings and skip thoughts for train validation and test sets
create_data:
	cd data/glove-embeddings && \
	wget https://nlp.stanford.edu/data/wordvecs/glove.6B.zip && \
	unzip glove.6B.zip && \
	rm glove.6B.zip glove.6B.200d.txt glove.6B.300d.txt glove.6B.50d.txt
	cd data/skip-thought && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/NQ9OT8Xxvdxn3wo/download -o skip-thoughts-embeddings_train.npy && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/10CivpGpg8O1Bfe/download -o skip-thoughts-embeddings_validation.npy && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/PKQm7YuCMsPhBv6/download -o skip-thoughts-embeddings_test.npy

run_bert:
	EPOCHS=5; \
	PERCENTAGE_SYNONYMS=0.2; \
	NETWORK=bidirectional-1024-1-1-True-lstm:highway-3; \
	LEARNING_RATE=2e-5; \
	NUM_ESTIMATORS=10; \
	bsub -n 2 -W 23:59 -R "rusage[mem=12000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python bert_sct.py --data_dir /cluster/home/sanagnos/NLU/project2/data --output_dir /scratch/$${USER}/output_dir --tfhub_cache_dir /scratch/$${USER}/tfhub_cache_dir --num_epochs $${EPOCHS} --learning_rate $${LEARNING_RATE} --num_estimators $${NUM_ESTIMATORS} --network $${NETWORK} --percentage_synonyms $${PERCENTAGE_SYNONYMS} --save_results_dir ./results_predictions_$${EPOCHS}_$${PERCENTAGE_SYNONYMS}_$${NETWORK}

run_all:
	run_bert

.PHONY: create_data run_bert run_all
