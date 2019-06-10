EPOCHS=5
PERCENTAGE_SYNONYMS=0
NETWORK=None
LEARNING_RATE=2e-5
NUM_ESTIMATORS=5

bsub -n 2 -W 23:59 -R "rusage[mem=12000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python bert_sct.py --data_dir /cluster/home/sanagnos/NLU/project2/data --output_dir /scratch/${USER}/output_dir --tfhub_cache_dir /scratch/${USER}/tfhub_cache_dir --num_epochs ${EPOCHS} --learning_rate ${LEARNING_RATE} --num_estimators ${NUM_ESTIMATORS} --network ${NETWORK} --percentage_synonyms ${PERCENTAGE_SYNONYMS} --save_results_dir ./results_predictions_${EPOCHS}_${PERCENTAGE_SYNONYMS}_${NETWORK}
