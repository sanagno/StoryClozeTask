module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
module load cudnn/7.2

bsub -n 2 -W 23:59 -R "rusage[mem=12000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python bert_sct.py --data_dir /cluster/home/sanagnos/NLU/project2/data --output_dir /scratch/sanagnos/output_dir --tfhub_cache_dir /scratch/sanagnos/tfhub_cache_dir --num_epochs 5 --learning_rate 2e-5 --num_estimators 15 --network bidirectional-1024-1-1-True-lstm:highway-3 
