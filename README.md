# ML_N-Gram_Model


## Introduction

This project seeks to predict Java token-based sequences by using an N-gram model. The model learns the probabilities of tokens sequences based on occurrences in the provided training data.

## Install Instructions
> [!NOTE]
> Sequentially follow these instructions.

First, clone our repo (in a desired location locally).
```
git clone https://github.com/ajarnold920/ML_N-Gram_Model.git
```
Move into the cloned repo using cd.
```
cd ML_N-Gram_Model
```
Next, we install our dependencies for the project.
```
pip install -r requirements.txt
```

## Running the Project

To run the project using the training set provided by our professor, use the command:
```
python ngrams.py M_teacher.txt
```

To run the project while only calculating perplexity on a smaller evaluation set for faster speed use the command:
```
python ngrams_fast_eval.py M_teacher.txt
```

This is especially useful if you run the project with a new, smaller corpus for testing purposes:
```
python ngrams_fast_eval.py yourtrainingset.txt
```

To calculate the perplexity on the evaluation and test sets for a saved model, use the command:
```
python ngrams_load_pkl.py savedmodel.pkl
```

To do the same but only on a smaller evaluation set for faster speed, use the command:
```
python ngrams_load_pkl_fast_eval.py savedmodel.pkl
```
