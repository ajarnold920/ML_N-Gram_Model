# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NL32O1AvNxC_tyenTwKveyG_wwjVweto
"""

import sys

# Check if arguments are provided
if len(sys.argv) > 1:
  corpus = sys.argv[1]

# Preprocessing

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

    #Example
code = """public static void main() { System.out.println("bau");}"""

lexer = JavaLexer()

tokens = [t[1] for t in lexer.get_tokens(code)]
print(tokens)
print(len(tokens))

#data = pd.read_csv("ghsajp1.csv", usecols=['Method Java'])
#data = pd.read_csv("test.csv", usecols=['Method Java'])
with open(corpus, 'r') as file:
    lines = file.readlines()
data = pd.DataFrame(lines, columns=['Method Java'])
#data["Method Java"] = data["Method Java"].replace("\\n","", regex=True)
#print(len(data))



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


# Example usage
#data = pd.DataFrame({
#    "Method Java": [
#        "public void setName(String name) { this.name = name; }",
#        "public String getName() { return this.name; }",
#        "public void processData() { System.out.println(\"Processing data\"); }",
#        "// This is a comment\npublic void processData() { /* Do something */ System.out.println(\"Done\"); }",
#        "public void doWork() { for(int i=0; i<10; i++) /* Do something */ System.out.println(i); }",
#    ]
#})

print("Initial dataset size:", len(data))
data = remove_duplicates(data)
print("After removing duplicates:", len(data))

data = filter_ascii_methods(data)
print("After filtering ASCII methods:", len(data))

data = remove_outliers(data)
print("After removing outliers:", len(data))

data = remove_boilerplate_methods(data)
print("After removing boilerplate methods:", len(data))

data = remove_comments_from_dataframe(data, "Method Java", "Java")
print("After cleaning comments:", len(data))

print(data.head())

#data["Method Java No Comments"].to_csv("bum.txt", index=False, header=False)
#with open('bum.txt', 'r') as infile, open('bum2.txt', 'w') as outfile:
#    for line in infile:
#        # Write the line to the output file, skipping the first character
#        outfile.write(line[1:])
#with open('bum2.txt', 'r') as infile, open('M_Student.txt', 'w') as outfile:
#    for line in infile:
#        # Write the line to the output file, skipping the first character
#        if(line != "\n"):
#          outfile.write(line)

from sklearn.model_selection import train_test_split

train_eval, test = train_test_split(data, test_size=0.2, random_state=42)
train, eval = train_test_split(train_eval, test_size=0.25, random_state=42)

#Creates an n-gram model from the given corpus and returns it
def nGram(corpus, n):
  model = pd.DataFrame(columns=['gram', 'correct', 'count'])
  corpus = corpus['Method Java No Comments']
  for line in corpus:
    words = line.split(' ')
    words = list(filter(None, words))
    for i in range(len(words) - n + 1):
      gram = ' '.join(words[i:i+n-1])
      if(i+n < len(words)):
        correct = words[i+n-1]
      else:
        correct = ''
      new_entry = {'gram': gram, 'correct': correct}

      # Check if the row exists
      mask = (model['gram'] == new_entry['gram']) & (model['correct'] == new_entry['correct'])

      if model[mask].shape[0] > 0:  # If exists
        model.loc[mask, 'count'] += 1
      else:  # If doesn't exist, add a new row
        new_entry['count'] = 1
        model = pd.concat([model, pd.DataFrame([new_entry])], ignore_index=True)

  return model

#model = nGram(train, 3)

#model

vocab = []
for line in data['Method Java No Comments']:
  words = line.split(' ')
  words = list(filter(None, words))
  for word in words:
    if word not in vocab:
      vocab.append(word)

vocabLength = len(vocab)

def predict(model, sequence):
  count = 0
  n = len(list(filter(None, model['gram'].iloc[0].split(' '))))
  words = sequence.split(' ')
  words = list(filter(None, words))
  gram = ' '.join(words[-n:])
  countnum = 1
  countdenom = vocabLength
  matches = model[model['gram'] == gram]
  prediction = ''
  for line in matches.itertuples():
    count += 1
    countdenom += line.count
    if line.count + 1 >= countnum:
      countnum = line.count + 1
      prediction = line.correct
  return (prediction, countnum/countdenom)

#testString = '( int'
#print(predict(model, testString))

import numpy

def perplexity(model, n, eval):
  perp = numpy.float128(0)
  count = 0
  corpus = eval['Method Java No Comments']
  # print(corpus)
  for line in corpus:
    words = line.split(' ')
    words = list(filter(None, words))
    for i in range(len(words) - n + 1):
      count+=1
      probnum = numpy.float128(1)
      probdenom = numpy.float128(vocabLength)
      gram = ' '.join(words[i:i+n-1])

      matches = model[model['gram'] == gram]
      for row in matches.itertuples():
        if gram == row.gram:
          probdenom += row.count

        if i+n == len(words):
          if '' == row.correct:
            probnum += row.count
        elif words[i+n] == row.correct:
          probnum += row.count


      # if(probdenom == 0):
      #   prob = 1

      # prob = numpy.float128(probnum/probdenom)
      prob = numpy.log(probnum/probdenom)

      perp += prob

  print(count)

  return numpy.exp(-perp/count)
  # return pow(perp, 1/count)

# Originally got 1586.9922687733
#perplexity(model, 3, eval)

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

#print(iterative_predict(model, ('@', 'Override'), 2))

n = [3, 5, 7]
models = []
perps = []
minPerp = 1000000000
minPerpIndex = 0
for i in range(len(n)):
  print("Working on " + str(n[i]))
  model = nGram(train, n[i])
  model.to_pickle()
  models.append(model)
  perp = perplexity(model, n[i], eval)
  perps.append(perp)
  print(str(n[i]) + "-Gram Perplexity: " + str(perp))
  if(perp < minPerp):
    minPerp = perp
    minPerpIndex = i
print("Best n: " + str(n[minPerpIndex]))

import json

model = models[minPerpIndex]

for i in range(100):
  rand = numpy.random.randint(0, len(test))
  method = test['Method Java No Comments'].iloc[rand]
  while("<" in method[:5] or method[0] == '"' or method[0] == "'"):
    rand = numpy.random.randint(0, len(test))
    method = test['Method Java No Comments'].iloc[rand]
  method = list(filter(None, method.split(' ')))
  seed = tuple(method[:n[minPerpIndex]-1])
  print(seed)
  prediction = iterative_predict(model, seed, len(method))
  dictionary = {
    "index": str(i),
    "prediction": prediction,
  }
  with open("sample.json", "a") as outfile:
    json.dump(dictionary, outfile)

#Train models (3, 5, 7, 9) maybe more using ghs data
#Calculate perplexity on evaluation set
#Test models on test set and make predictions

#Train new models on his data
#Calculate perplexity on SAME evaluation set from the ghs data
#Test models on SAME test set from ghs data make predictions

#save models with pkl
#60-80% in training set

#Model keeps predicting until parenthesis are balanced or probabilities very close to 0
#Predict from the first n of the tokens
#If we want, we can "cheat" and use the length of the method and generate up to that
