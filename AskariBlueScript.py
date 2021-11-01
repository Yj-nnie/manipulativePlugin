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
from nltk.tokenize import RegexpTokenizer
import joblib
import sys

def read_email(path):
  with open(path, 'rb') as f:
    email = f.read().decode('utf8','ignore')
  return email

def load_model(path):
  model = joblib.load(path)
  return model

def preprocess(x):

  # remove extra new lines
  x = re.sub(r'\n+', ' ', x)
    
  # remove extra white spaces
  x = re.sub(r'\s+', ' ', x)
    
  #remove <html> tags
  p = re.compile(r'<.*?>')
  x = p.sub('', x)

  # remove hyperlinks
  x=re.sub(r'http\S+', '', x)

  
  return x

def word_count(emailtext):

  tokenizer = RegexpTokenizer(r'\w+')
  
  tokens = tokenizer.tokenize(emailtext)
  
  return len(tokens)

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
  grammarTag['wordcount'] = word_count(emailtext)

  # total number of punctuation
  count = lambda l1,l2: sum([1 for x in l1 if x in l2])
  grammarTag['totalPunctuation']= count(emailtext,set(string.punctuation)) 

  #total number of sentences
  regex = r"[A-Z][\w\s\d\,\'\;\:\"\(\)\[\]\%\$\!\?]*(\.)"
  grammarTag['totalSentence']=len(list(re.finditer(regex, emailtext)))

  # total characters without space
  grammarTag['totalCharacters'] = len(emailtext) - emailtext.count(" ")

  return grammarTag

def phraseFeatures(emailtext, features_dict):

  #complexity
  features_dict['avg_sentenceLength'] = features_dict['wordcount']/features_dict['totalSentence']
  features_dict['avg_wordLength'] = features_dict['totalCharacters']/features_dict['wordcount']
  features_dict['pausality'] = features_dict['totalPunctuation']/features_dict['totalSentence'] 

  #uncertainty
  features_dict['modifier'] = features_dict['Adjectives'] + features_dict['Adverbs']
  features_dict['Uncertainty'] = features_dict['modifier'] / features_dict['wordcount']
  features_dict['nonimmediacy'] = features_dict['PRP'] / features_dict['wordcount']

  #expressiveness
  features_dict['Expressiveness'] = features_dict['modifier'] / (features_dict['Nouns'] + features_dict['Verbs'])
  
  #authority
  number_of_you = pd.Series([emailtext]).str.count(r'\b(you[r]*)\b', flags=re.I).iat[0]
  features_dict['Authority'] =  number_of_you / features_dict['wordcount']
  


  return features_dict


def sentimentScore(sentence, features_dict):

 # Create a SentimentIntensityAnalyzer object.
    analyzer = SentimentIntensityAnalyzer()
 #
    sentiment_dict = analyzer.polarity_scores(sentence)

    features_dict['neg'] = sentiment_dict['neg']
    features_dict['neu'] = sentiment_dict['neu']
    features_dict['pos'] = sentiment_dict['pos']
    features_dict['scoreTag'] = sentimentTag(sentiment_dict['compound'])

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

def wordswithCaps(emailtext, feature_dict):

  upper_case = 0
  num_words = word_count(emailtext)

  for word in RegexpTokenizer(r'\w+').tokenize(emailtext): #list of tokens 
    if word.isupper():
      upper_case += 1

  percentage = upper_case/num_words
  feature_dict['caps percentage'] = percentage

  return feature_dict

def transform_email(emailtext): #obtain all features
  
  #Primary features: add grammar tags and sentence characteristic tags
  primary_feature = extractGrammar(posTag_email_one(emailtext))
  primary_feature = addPhraseCharacteristics(emailtext,primary_feature)

  #Features based on primary features
  phrase_features = phraseFeatures(emailtext, primary_feature)
  
  #sentiment features
  sentiment_features = sentimentScore(emailtext, phrase_features)

  #all Caps features
  final_feature_dict = wordswithCaps(emailtext, sentiment_features)

  return sentiment_features

def predict(features_dict, model):

  proba = model.predict_proba(features_dict)
  
  predicted_class = model.classes_[(np.argmax(proba))] 
  return {
      "class": predicted_class,
      "proba": np.max(proba)
  }


def main(email_path, model_path):
  model = load_model(model_path)
  email = preprocess(read_email(email_path))
  features = transform_email(email)

  features_names = ['avg_sentenceLength', 'avg_wordLength', 'pausality',  'Uncertainty', 'nonimmediacy', 'Expressiveness', 'Authority', 'neg', 'neu', 'pos', 'scoreTag', 'caps percentage'] #list
  features_list = [features[fn] for fn in features_names]
  pred = predict(np.array([features_list]), model)

  print(pred)
  return pred


if __name__ == "__main__":
  email_path = sys.argv[2]
  model_path = sys.argv[1]

  prediction = main(email_path,model_path)



#emailtext = preprocess (read_email('/content/gdrive/MyDrive/Github/ManipulativePlugin/Sample files /m1.txt'))
#model = joblib.load('/content/gdrive/MyDrive/Github/ManipulativePlugin/Models/V6_7_SVMTrainModel Second')
#features = transform_email(emailtext)

#print(features)
#features_names = ['avg_sentenceLength', 'avg_wordLength', 'pausality',  'Uncertainty', 'nonimmediacy', 'Expressiveness', 'Authority', 'neg', 'neu', 'pos', 'scoreTag', 'caps percentage'] #list
#features_list = [features[fn] for fn in features_names]
#predict = predict(np.array([features_list]), model)

#print(predict)

