# Create skip thoughts embeddings.
# This should be done only once. After, use the created embeddings for faster processing.

from sklearn.utils import shuffle
from skip_thoughts import load_model, Encoder
import os
import nltk
import pandas as pd
import numpy as np


def get_batches(iterable, batch_size=64, do_shuffle=True):
    """
    Generate batches
    Parameters
    ----------
    iterable: list
        data to generate batches for
    batch_size: int
    do_shuffle: bool
        Whether to shuffle in each epoch
    """
    if do_shuffle:
        iterable = shuffle(iterable)

    length = len(iterable)
    for ndx in range(0, length, batch_size):
        iterable_batch = iterable[ndx: min(ndx + batch_size, length)]
        yield iterable_batch


CWD='/cluster/project/infk/courses/machine_perception_19/Sasglentamekaiedo'


# Directory to download the pretrained models to.
PRETRAINED_MODELS_DIR=os.path.join(CWD, '/skip_thoughts/pretrained/')

os.mkdir(PRETRAINED_MODELS_DIR)
os.chdir(PRETRAINED_MODELS_DIR)

# Download and extract the unidirectional model.
os.system("wget http://download.tensorflow.org/models/skip_thoughts_uni_2017_02_02.tar.gz")
os.system("tar -xvf skip_thoughts_uni_2017_02_02.tar.gz")
os.system("rm skip_thoughts_uni_2017_02_02.tar.gz")

# Download and extract the bidirectional model.
os.system("wget http://download.tensorflow.org/models/skip_thoughts_bi_2017_02_16.tar.gz")
os.system("tar -xvf skip_thoughts_bi_2017_02_16.tar.gz")
os.system("rm skip_thoughts_bi_2017_02_16.tar.gz")

DIRECTORY='/cluster/project/infk/courses/machine_perception_19/Sasglentamekaiedo/skip_thoughts_npy'

os.system("wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt --directory-prefix=%s"%DIRECTORY)
os.system("wget http://www.cs.toronto.edu/~rkiros/models/utable.npy --directory-prefix=%s"%DIRECTORY)
os.system("wget http://www.cs.toronto.edu/~rkiros/models/btable.npy --directory-prefix=%s"%DIRECTORY)
os.system("wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz --directory-prefix=%s"%DIRECTORY)
os.system("wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl --directory-prefix=%s"%DIRECTORY)
os.system("wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz --directory-prefix=%s"%DIRECTORY)
os.system("wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl --directory-prefix=%s"%DIRECTORY)

model = load_model()
encoder = Encoder(model)

# download nltk punkt
nltk.download('punkt')

# get pandas dataframe columns that containt the data
names = ['InputSentence' + str(i) for i in [1, 2, 3, 4]]
names.append('RandomFifthSentenceQuiz1')
names.append('RandomFifthSentenceQuiz2')

data_directory = './data/'
data_val = pd.read_csv(os.path.join(data_directory, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'),
                       header='infer')

all_sentences = data_val[names].values.reshape(-1)

all_sentences_encoded = []

# create skip thoughts embeddings
# used a relatively large batch size that does not lead to GPU out of memory problems
i = 0
batch_size = 2048*8
for sentences_batch in get_batches(all_sentences, batch_size=batch_size, do_shuffle=False):
    print('new batch', i)
    i += batch_size
    all_sentences_encoded.append(encoder.encode(sentences_batch, verbose=False))

# transform into previous accepted state
all_sentences_encoded_np = np.concatenate(all_sentences_encoded, axis=0)
all_sentences_encoded_np = all_sentences_encoded_np.reshape((-1, 6, 4800))

save_file = '/cluster/project/infk/courses/' \
            'machine_perception_19/Sasglentamekaiedo/skip-thoughts-embbedings_validation.npy'

np.save(save_file, all_sentences_encoded_np)
