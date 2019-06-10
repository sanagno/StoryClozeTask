# for hyperparameters selection options run relative python script with the flag -h (--help)
help:
	@echo  'Makefile options:'
	@echo  '  setup           	 - Install requirements in the virtual environment.'
	@echo  '                    	   Loads skip-thoughts embeddings and glove embeddings.'
	@echo  '  run_bert	         - Runs bert classifier. Creates best predictions for '
	@echo  '                           labeled and unlabeled test set. '
	@echo  '  run_simple      	 - Runs models of paper (https://www.aclweb.org/anthology/N18-2015).'
	@echo  ' 	                   A Simple and Effective Approach to the Story Cloze Test. '
	@echo  '  run_word_based      	 - BiLSTM encoder and decoder with attention.'
	@echo  '  run_glove_and_sent2vec - Runs model that combines glove and sent2vec representations.'
	@echo  '  run_sentiment_analysis - Runs model that predicts polarity of the fifth sentence.'
	@echo  '  run_predict_context  	 - Runs model that predicts the context of the fifth sentence.'
	@echo  ''

# downloads glove 100d embeddings and skip thoughts for train validation and test sets
setup: data/glove-embeddings/glove.6B.100d.txt data/skip-thoughts/skip-thoughts-embeddings_train.npy data/skip-thoughts/skip-thoughts-embeddings_validation.npy data/skip-thoughts/skip-thoughts-embeddings_test.npy
	pip install -r requirements.txt 

data/glove-embeddings/glove.6B.100d.txt:
	cd data/glove-embeddings && \
        wget https://nlp.stanford.edu/data/wordvecs/glove.6B.zip && \
        unzip glove.6B.zip && \
        rm glove.6B.zip glove.6B.200d.txt glove.6B.300d.txt glove.6B.50d.txt

data/skip-thoughts/skip-thoughts-embeddings_train.npy:
	cd data/skip-thoughts && \
        wget --no-check-certificate https://polybox.ethz.ch/index.php/s/NQ9OT8Xxvdxn3wo/download -O skip-thoughts-embeddings_train.npy

data/skip-thoughts/skip-thoughts-embeddings_validation.npy:
	cd data/skip-thoughts && \
        wget --no-check-certificate https://polybox.ethz.ch/index.php/s/10CivpGpg8O1Bfe/download -O skip-thoughts-embeddings_validation.npy

data/skip-thoughts/skip-thoughts-embeddings_test.npy:
	cd data/skip-thoughts && \
        wget --no-check-certificate https://polybox.ethz.ch/index.php/s/PKQm7YuCMsPhBv6/download -O skip-thoughts-embeddings_test.npy


# runs: bert
run_bert:
	EPOCHS=5; \
	DATA_DIR="./data/ROCStories"; \
	PERCENTAGE_SYNONYMS=0.2; \
	NETWORK=bidirectional-1024-1-1-True-lstm:highway-3; \
	LEARNING_RATE=2e-5; \
	NUM_ESTIMATORS=10; \
	bsub -n 2 -W 03:59 -R "rusage[mem=12000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" -o lsf_bert.out python ./src/bert/bert_sct.py --data_dir $${DATA_DIR} --output_dir /scratch/$${USER}/output_dir --tfhub_cache_dir /scratch/$${USER}/tfhub_cache_dir --num_epochs $${EPOCHS} --learning_rate $${LEARNING_RATE} --num_estimators $${NUM_ESTIMATORS} --network $${NETWORK} --percentage_synonyms $${PERCENTAGE_SYNONYMS} --save_results_dir ./bert_predictions_$${EPOCHS}_$${PERCENTAGE_SYNONYMS}_$${NETWORK}

# runs: A Simple and Effective Approach to the Story Cloze Test
# https://www.aclweb.org/anthology/N18-2015
run_simple:
	LOG_PATH="./log_path"; \
	DATA_DIR="./data/ROCStories"; \
	EPOCHS=10; \
	BATCH_SIZE=32; \
	LEARNING_RATE=1e-3; \
	UNITS=4800; \
	TRAIN_ON_VALIDATION=1; \
	MODE="LS-skip"; \
	if [ "$$TRAIN_ON_VALIDATION" -eq 0 ]; then \
	    MEMORY=18000; \
	else \
	    MEMORY=5000; \
	fi; \
	bsub -o lsf_simple_$${TRAIN_ON_VALIDATION}_$${MODE}_$${LEARNING_RATE}.out -n 2 -W 03:59 -R "rusage[mem=$${MEMORY},ngpus_excl_p=1]" python ./src/simple_effective_approach.py --data_dir /cluster/home/sanagnos/NLU/project2/data --log_path /scratch/$${USER}/log_path --num_epochs $${EPOCHS} --learning_rate $${LEARNING_RATE} --verbose True --train_on_validation $${TRAIN_ON_VALIDATION} --mode $${MODE} --batch_size $${BATCH_SIZE} --log_path $${LOG_PATH} --data_dir $${DATA_DIR}

run_glove_and_sent2vec:
	bsub -n 2 -W 4:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -o lsf_run_glove_and_sent2vec.out python src/run_Glove_and_Sent2vec_model.py

run_word_based:
	bsub -n 2 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" -o lsf_run_word_based.out python src/word_based_model.py

run_sentiment_analysis:
	bsub -n 1 -W 8:00 -R "rusage[mem=64192, ngpus_excl_p=1]" -o lsf_run_sentiment_analysis.out python src/sentimentLSTM.py

run_predict_context:
	bsub -n 1 -W 8:00 -R "rusage[mem=64192, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" -o lsf_run_predict_context.out python src/contextLSTM.py

run_all: run_bert run_simple run_glove_and_sent2vec run_word_based run_sentiment_analysis run_predict_context

.PHONY: help setup run_bert run_simple run_glove_and_sent2vec run_word_based run_sentiment_analysis run_predict_context run_all
