# StoryClozeTask

This project is part of the Natural Languange Understanding course (2019) in ETH.

Task: Find the best ending for a story (https://arxiv.org/abs/1604.01696).


Team: 25 

| Name  | Email |
| ------------- | ------------- |
| Ioannis Sachinoglou  | saioanni@student.ethz.ch  |
| Adamos Solomou  | solomoua@student.ethz.ch  |
| Anagnostidis Sotiris  | sanagnos@student.ethz.ch  |
| Georgios Vasilakopoulos  | gvasilak@student.ethz.ch  |

## Project structure

    .
    ├── data                                # Datasets (train, validation, test) 
    ├── results                             # Predictions for unlabeled test set
    ├── src                                 # Source files
    │   ├── bert                            # Files concerning running BERT classifier for the task.
    │   ├── create_skip_thoughts_embeddings # Script to create skip thoughts embeddings.
    └── README.md
    
    
## Getting Started

### Prerequisites

- Install Python 3.6+
- Install requirements
  ```
  make requirements
  ```
- Some experiments require skip thoughts embeddings as specified in the paper Skip-Thought Vectors (https://arxiv.org/abs/1506.06726 ,https://github.com/ryankiros/skip-thoughts). For time saving purposes these have been precomputed and are publicly available at https://polybox.ethz.ch/index.php/s/X3GsRxeIhATdt8J. They files saved have the form of a numpy array with a shape [num_samples, num_sentences, skip_thought_embeddings_size]. For the training set, each story has a total of 5 sentences, while for the validation and test set each story has 6 sentences (corresponding to the two possible endings). To install:
  ```
  mkdir skip-thoughts
  cd skip-thoughts
  wget --no-check-certificate https://polybox.ethz.ch/index.php/s/NQ9OT8Xxvdxn3wo/download -o skip-thoughts-embeddings_train.npy
  wget --no-check-certificate https://polybox.ethz.ch/index.php/s/10CivpGpg8O1Bfe/download -o skip-thoughts-embeddings_validation.npy
  wget --no-check-certificate https://polybox.ethz.ch/index.php/s/PKQm7YuCMsPhBv6/download -o skip-thoughts-embeddings_test.npy
  ```

## Run experiments

Create specific running scripts (should merge cases that implement the nlu model)
