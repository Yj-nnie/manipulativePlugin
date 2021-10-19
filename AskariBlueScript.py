import pandas as pd
import re
import numpy as np
import string
import nltk
nltk.download('all')
from collections import Counter
from nltk import word_tokenize, pos_tag, pos_tag_sents

def read_email(path):
  with open(path, 'r') as f:
    email = f.read()
  #print(email)
  return email

def preprocess(x):

    # remove extra new lines
    x = re.sub(r'\n+', ' ', x)
    
    # remove extra white spaces
    x = re.sub(r'\s+', ' ', x)
    
    #getting rid of html/formatting tags -----> Decision MADE?
    #this step does not exist in my training data because there were no HTML tags
    #but text file from Askari Blue does have html tags.
    p = re.compile(r'<.*?>')
    x = p.sub('', x)

    #print(x)
    return x

#Use pos_tag to get grammatical tags
def posTag_email_one(email):
    tagged_text = pos_tag(word_tokenize(email))
    return tagged_text 

def extractGrammar(x): 
    # Updated calculate the tags I need 
    tag_count_data = Counter([tag[1] for tag in x])
   
    #Convert the Counter object to dict
    tag_count_dict = dict(tag_count_data)
    
    #select Tags that I need
    pos_columns = ['PRP','MD','JJ','JJR','JJS','RB','RBR','RBS', 'NN', 'NNS','VB', 'VBS', 'VBG','VBN','VBP','VBZ']
    for pos in pos_columns:
      if pos not in tag_count_dict.keys():
         tag_count_dict[pos] = 0

    #return dictionary item
    email_tag = {}
    
    email_tag['PRP'] = tag_count_dict['PRP']
    email_tag['MD'] = tag_count_dict['MD']
    email_tag['Adjecitves'] = tag_count_dict['JJ'] + tag_count_dict['JJR'] + tag_count_dict['JJS']
    email_tag['Adverbs'] = tag_count_dict['RB'] + tag_count_dict['RBR'] + tag_count_dict['RBS']
    email_tag['Nouns'] = tag_count_dict['NN'] + tag_count_dict['NNS']
    email_tag['Verbs'] = tag_count_dict['VB'] + tag_count_dict['VBS'] + tag_count_dict['VBG'] + tag_count_dict['VBN'] + tag_count_dict['VBP'] + tag_count_dict['VBZ']

    return email_tag

emailtext = preprocess (read_email('/content/gdrive/MyDrive/Github/ManipulativePlugin/Sample files /m1.txt'))
x_grammarTag = extractGrammar(posTag_email_one(emailtext))
print(x_grammarTag)
