# for hyperparameters selection options run relative python script with the flag -h (--help)
help:
	@echo  'Makefile options:'
	@echo  '  load_data       - Loads skip-thoughts embeddings and glove embeddings'
	@echo  '  run_bert        - Runs bert classifier. Creates best predictions for '
	@echo  '                    labeled and unlabeled test set. '
	@echo  '  run_simple      - Runs models of paper (https://www.aclweb.org/anthology/N18-2015).'
	@echo  '                    A Simple and Effective Approach to the Story Cloze Test. '
	@echo  ''

# downloads glove 100d embeddings and skip thoughts for train validation and test sets
load_data:
	cd data/glove-embeddings && \
	wget https://nlp.stanford.edu/data/wordvecs/glove.6B.zip && \
	unzip glove.6B.zip && \
	rm glove.6B.zip glove.6B.200d.txt glove.6B.300d.txt glove.6B.50d.txt
	cd data/skip-thought && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/NQ9OT8Xxvdxn3wo/download -o skip-thoughts-embeddings_train.npy && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/10CivpGpg8O1Bfe/download -o skip-thoughts-embeddings_validation.npy && \
	wget --no-check-certificate https://polybox.ethz.ch/index.php/s/PKQm7YuCMsPhBv6/download -o skip-thoughts-embeddings_test.npy

# runs: bert
run_bert:
	EPOCHS=5; \
	DATA_DIR="./data/ROCStories"; \
	PERCENTAGE_SYNONYMS=0.2; \
	NETWORK=bidirectional-1024-1-1-True-lstm:highway-3; \
	LEARNING_RATE=2e-5; \
	NUM_ESTIMATORS=10; \
	bsub -n 2 -W 03:59 -R "rusage[mem=12000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python ./src/bert/bert_sct.py --data_dir $${DATA_DIR} --output_dir /scratch/$${USER}/output_dir --tfhub_cache_dir /scratch/$${USER}/tfhub_cache_dir --num_epochs $${EPOCHS} --learning_rate $${LEARNING_RATE} --num_estimators $${NUM_ESTIMATORS} --network $${NETWORK} --percentage_synonyms $${PERCENTAGE_SYNONYMS} --save_results_dir ./bert_predictions_$${EPOCHS}_$${PERCENTAGE_SYNONYMS}_$${NETWORK}

# runs: A Simple and Effective Approach to the Story Cloze Test
# https://www.aclweb.org/anthology/N18-2015
run_simple:
	LOG_PATH="./log_path"; \
	DATA_DIR="./data/ROCStories"; \
	EPOCHS=10; \
	BATCH_SIZE=32; \
	LEARNING_RATE=1e-3; \
	UNITS=4800; \
	TRAIN_ON_VALIDATION=0; \
	MODE="LS-skip"; \
	if [ "$$TRAIN_ON_VALIDATION" -eq 0 ]; then \
	    MEMORY=18000; \
	else \
	    MEMORY=5000; \
	fi; \
	bsub -o lsf_simple_$${TRAIN_ON_VALIDATION}_$${MODE}_$${LEARNING_RATE}.out -n 2 -W 03:59 -R "rusage[mem=$${MEMORY},ngpus_excl_p=1]" python ./src/simple_effective_approach.py --data_dir /cluster/home/sanagnos/NLU/project2/data --log_path /scratch/$${USER}/log_path --num_epochs $${EPOCHS} --learning_rate $${LEARNING_RATE} --verbose True --train_on_validation $${TRAIN_ON_VALIDATION} --mode $${MODE} --batch_size $${BATCH_SIZE} --log_path $${LOG_PATH} --data_dir $${DATA_DIR}

run_sentiment:
	python src/sentimentLSTM.py

run_predict_context:
	python src/contextLSTM.py

run_all:
	run_bert

.PHONY: load_data run_bert run_simple run_all
