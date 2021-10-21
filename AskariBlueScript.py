import pandas as pd
import re
import numpy as np
import string
import nltk
nltk.download('all')
from collections import Counter
from nltk import word_tokenize, pos_tag, pos_tag_sents
#pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    email_tag['Adjectives'] = tag_count_dict['JJ'] + tag_count_dict['JJR'] + tag_count_dict['JJS']
    email_tag['Adverbs'] = tag_count_dict['RB'] + tag_count_dict['RBR'] + tag_count_dict['RBS']
    email_tag['Nouns'] = tag_count_dict['NN'] + tag_count_dict['NNS']
    email_tag['Verbs'] = tag_count_dict['VB'] + tag_count_dict['VBS'] + tag_count_dict['VBG'] + tag_count_dict['VBN'] + tag_count_dict['VBP'] + tag_count_dict['VBZ']

    return email_tag

# Add text characteristics to existing dictionary
def addPhraseCharacteristics(emailtext,grammarTag):

  #wordCount
  grammarTag['wordcount'] = len(re.findall("[a-zA-Z_]+", emailtext))

  # total number of punctuation
  count = lambda l1,l2: sum([1 for x in l1 if x in l2])
  grammarTag['totalPunctuation']= count(emailtext,set(string.punctuation)) 

  # totalDots = at least one because there is at least one phrase in an email
  grammarTag['totalDots'] = emailtext.count('[.]') 
  if grammarTag['totalDots'] == 0:
     grammarTag['totalDots'] = 1

  # total characters without space
  grammarTag['totalCharacters'] = len(emailtext) - emailtext.count(" ")

  return grammarTag

def phraseFeatures(emailtext, features_dict):

  #complexity
  features_dict['avg_sentenceLength'] = features_dict['wordcount']/features_dict['totalDots']
  features_dict['avg_wordeLength'] = features_dict['totalCharacters']/features_dict['wordcount']
  features_dict['pausality'] = features_dict['totalPunctuation']/features_dict['totalDots'] 

  #uncertainty
  features_dict['modifier'] = features_dict['Adjectives'] + features_dict['Adverbs']
  features_dict['Uncertainty'] = features_dict['modifier'] / features_dict['wordcount']
  features_dict['nonimmediacy'] = features_dict['PRP'] / features_dict['wordcount']

  #expressiveness
  features_dict['Expressiveness'] = features_dict['modifier'] / (features_dict['Nouns'] + features_dict['Verbs'])
  
  #authority
  features_dict['Authority'] = pd.Series([emailtext]).str.count(r'\b(you[r]*)\b', flags=re.I).iat[0]

  #neaten up needed features
  key_to_remove =("PRP", "MD","Adjectives","Adverbs","Nouns","Verbs","wordcount","totalPunctuation","totalDots","totalCharacters","modifier")
  for k in key_to_remove:
    if k in features_dict:
      del features_dict[k]


  return features_dict


def sentimentScore(sentence, features_dict):

 # Create a SentimentIntensityAnalyzer object.
    analyzer = SentimentIntensityAnalyzer()
 #
    sentiment_dict = analyzer.polarity_scores(sentence)

    features_dict['neg'] = sentiment_dict['neg']
    features_dict['neu'] = sentiment_dict['neu']
    features_dict['pos'] = sentiment_dict['pos']
    features_dict['compound'] = sentiment_dict['compound']
    features_dict['scoreTag'] = sentimentTag(features_dict['compound'])

    return features_dict 


def sentimentTag(compoundScore):

      if compoundScore >= 0.05 :
        #positive
        return 3
 
      elif compoundScore <= - 0.05 :
        #negative
        return 1
 
      else :
        #neutral
        return 2


def transform_email(emailtext): #obtain all features
  
  #Primary features: add grammar tags and sentence characteristic tags
  primary_feature = extractGrammar(posTag_email_one(emailtext))
  primary_feature = addPhraseCharacteristics(emailtext,primary_feature)

  #Features based on primary features
  phrase_features = phraseFeatures(emailtext, primary_feature)
  
  #sentiment features
  sentiment_features = sentimentScore(emailtext, phrase_features)

  return sentiment_features


emailtext = preprocess (read_email('/content/gdrive/MyDrive/Github/ManipulativePlugin/Sample files /m1.txt'))

features = transform_email(emailtext)

print(features)