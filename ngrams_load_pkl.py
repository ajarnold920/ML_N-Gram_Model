# -*- coding: utf-8 -*-

import sys

# Check if arguments are provided
if len(sys.argv) > 1:
  pkl = sys.argv[1]

import re
import pandas as pd

def remove_duplicates(data):
    """Remove duplicate methods based on method content.
      Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Java", keep="first")


def filter_ascii_methods(data):
    """Filter methods to include only those with ASCII characters."""
    #data = data[data["Method Java"].apply(lambda x: all(ord(char) < 128 for char in x))]
    data = data[data["Method Java"].apply(lambda x: isinstance(x, str) and all(ord(char) < 128 for char in x))]
    return data


# Three Approaches:
# 	1.	Data Distribution-Based Filtering: We eliminate outliers by analyzing the original data distribution, as demonstrated below.
# 	2.	Literature-Driven Filtering: We follow best practices outlined in research, such as removing methods exceeding 512 tokens in length.
# 	3.	Hybrid Approach: We combine elements from both the distribution-based and literature-driven methods.

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length."""
    method_lengths = data["Method Java"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token


with open('M_student.txt', 'r') as file:
    lines = file.readlines()
train = pd.DataFrame(lines, columns=['Method Java'])
with open('eval.txt', 'r') as file:
    lines = file.readlines()
eval = pd.DataFrame(lines, columns=['Method Java No Comments'])
with open('test.txt', 'r') as file:
    lines = file.readlines()
test = pd.DataFrame(lines, columns=['Method Java No Comments'])

def remove_boilerplate_methods(data):
    """Remove boilerplate methods like setters and getters."""
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data["Method Java"].apply(lambda x: bool(boilerplate_regex.search(x)))]
    return data


def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Removes comments from Java methods in a DataFrame and adds a new column with cleaned methods.

    Args:
        df (pd.DataFrame): DataFrame containing the methods.
        method_column (str): Column name containing the raw Java methods.
        language (str): Programming language for the lexer (e.g., 'java').

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Java Method No Comments'.
    """
    # Define a function to remove comments from a single method
    def remove_comments(code):
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        # Filter out comments using a lambda function
        clean_code = ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))


        return clean_code

    # Apply the function to the specified column and add a new column with the results
    df["Method Java No Comments"] = df[method_column].apply(remove_comments)
    return df


print("Initial dataset size:", len(train))
train = remove_duplicates(train)
print("After removing duplicates:", len(train))

train = filter_ascii_methods(train)
print("After filtering ASCII methods:", len(train))

train = remove_outliers(train)
print("After removing outliers:", len(train))

train = remove_boilerplate_methods(train)
print("After removing boilerplate methods:", len(train))

train = remove_comments_from_dataframe(train, "Method Java", "Java")
print("After cleaning comments:", len(train))

vocab = set([])
for line in train['Method Java No Comments']:
  words = line.split(' ')
  words = list(filter(None, words))
  for word in words:
    if word not in vocab:
      vocab.add(word)

for line in eval['Method Java No Comments']:
  words = line.split(' ')
  words = list(filter(None, words))
  for word in words:
    if word not in vocab:
      vocab.add(word)

for line in test['Method Java No Comments']:
  words = line.split(' ')
  words = list(filter(None, words))
  for word in words:
    if word not in vocab:
      vocab.add(word)

vocab = list(vocab)
vocabLength = len(vocab)

def predict(model, sequence):
  n = len(list(filter(None, model['gram'].iloc[0].split(' '))))
  words = sequence.split(' ')
  words = list(filter(None, words))
  gram = ' '.join(words[-n:])
  countnum = 1
  countdenom = vocabLength
  new_entry = {'gram': gram}
  matches = (model['gram'] == new_entry['gram'])
  prediction = ''
  if model[matches].shape[0] > 0:  # If gram exists
    preds = model.loc[matches, 'correct'].iloc[0]
    for line in preds:
        countdenom += line[1]
    if line[1] + 1 >= countnum:
      countnum = line[1] + 1
      prediction = line[0]
  return (prediction, countnum/countdenom)


import numpy

def perplexity(model, n, eval):
  perp = 0
  count = 0
  corpus = eval['Method Java No Comments']
  for line in corpus:
    words = line.split(' ')
    words = list(filter(None, words))
    for i in range(len(words) - n + 1):
      count += 1
      probnum = 1
      probdenom = vocabLength
      gram = ' '.join(words[i:i+n-1])

      new_entry = {'gram': gram}
      matches = (model['gram'] == new_entry['gram'])
      if model[matches].shape[0] > 0:
        preds = model.loc[matches, 'correct'].iloc[0]
        for row in preds:
          probdenom += row[1]
          if i+n == len(words):
            if '' == row[0]:
              probnum += row[1]
          elif words[i+n] == row[0]:
            probnum += row[1]


      prob = numpy.log(probnum/probdenom)

      perp += prob

  return numpy.exp(-perp/count)


def iterative_predict(model, seed, functionLength):
  words = [item for item in seed]
  n = len(list(filter(None, model['gram'].iloc[0].split(' '))))
  if(len(words) < n):
    return None
  predicting = True
  prediction = []
  while(predicting):
    prediction.append(predict(model, ' '.join(words[-n:])))
    words.append(prediction[-1][0])
    if(prediction[-1][0] == '' or len(words) >= functionLength):
      predicting = False
  return prediction

model = pd.read_pickle(pkl)
print(model.head())
n = len(list(filter(None, model['gram'].iloc[0].split(' ')))) + 1
print("Calculating Perplexity")
perp = perplexity(model, n, eval)
print(str(n) + "-Gram Perplexity: " + str(perp))

import json

print("Calculating Test Set Perplexity")
perptest = perplexity(model, n, test)
print("Test Perplexity: " + str(perptest))

for i in range(100):
  count = 1
  numpy.random.seed(i)
  rand = numpy.random.randint(0, len(test))
  method = test['Method Java No Comments'].iloc[rand]
  while("<" in method[:5] or method[0] == '"' or method[0] == "'"):
    numpy.random.seed(i + (100 * count))
    count += 1
    rand = numpy.random.randint(0, len(test))
    method = test['Method Java No Comments'].iloc[rand]
  method = list(filter(None, method.split(' ')))
  seed = tuple(method[:n-1])
  prediction = iterative_predict(model, seed, len(method))
  dictionary = {
    "index": str(i),
    "prediction": prediction,
  }
  with open("results_" + pkl[:-4] + "_model" + ".json", "a") as outfile:
    json.dump(dictionary, outfile)

