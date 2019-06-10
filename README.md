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
    ├── data                                
    │   ├── glove-embeddings                # 100d glove embeddings 
    │   ├── ROCStories                      # Datasets (train, validation, test)
    │   ├── skip-thoughts                   # Skip thoughts embeddings for (train, validation, test)
    ├── results                             # Predictions for unlabeled test set
    ├── src                                 # Source files
    │   ├── bert                            # Files concerning running BERT classifier for the task.
    │   ├── create_skip_thoughts_embeddings # Script to create skip thoughts embeddings.
    └── README.md
    
    
## Getting Started

### Prerequisites

- Install Python 3.6+
- Load modules and create virtual environment (works when running on eth leonhard cluster):
  ```
  source initialize.sh
  ```
- Install requirements and skip thought embeddings. Some experiments require skip thoughts embeddings as specified in the paper Skip-Thought Vectors (https://arxiv.org/abs/1506.06726 ,https://github.com/ryankiros/skip-thoughts). For time saving purposes these have been precomputed and are publicly available at https://polybox.ethz.ch/index.php/s/X3GsRxeIhATdt8J. They files saved have the form of a numpy array with a shape [num_samples, num_sentences, skip_thought_embeddings_size]. For the training set, each story has a total of 5 sentences, while for the validation and test set each story has 6 sentences (corresponding to the two possible endings). To install:
  ```
  make setup
  ```

## Run experiments

Create specific running scripts (should merge cases that implement the nlu model)
