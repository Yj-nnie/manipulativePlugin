import pandas as pd
import re
import numpy as np
import string
import nltk
nltk.download('all')
from collections import Counter
from nltk import word_tokenize, pos_tag, pos_tag_sents

def preprocess(x):

    # remove extra new lines
    x = re.sub(r'\n+', ' ', x)
    
    # remove extra white spaces
    x = re.sub(r'\s+', ' ', x)
    
    return x

def posTag_email(email):
    
    text = email.tolist()
    tagged_text = pos_tag_sents(map(word_tokenize, text))
    return tagged_text 


# read file into pandas dataFrame
emailtext = pd.read_csv('/content/gdrive/MyDrive/Github/ManipulativePlugin/Sample files /m1.txt', sep='\n', header=None)[0].str.cat()
emails = pd.DataFrame([emailtext], columns=['text'])

#Slightly preprocess incoming emails
emails['text'] = emails['text'].replace(np.nan, '')
emails.loc[:,'text'] = emails.loc[:, 'text'].map(preprocess)

#Add Pos_tag for 
emails["POS_Tag"] = emails.apply(posTag_email)
print(emails)

#Compute email text characteristics for Text-based features
# Updated calculate the tags I need 
###emails=emails.apply(extractGrammar, axis = 1)
### print(emails)

#######################Compute email text characteristics for Text-based features ----- this could totally go into a function but HOW???????######################
# Updated calculate the tags I need 

tag_count_data = pd.DataFrame(emails['POS_Tag'].map(lambda x: Counter(tag[1] for tag in x)).to_list())

#trying to have the POS taggs printed out for each data. 
emails = pd.concat([emails, tag_count_data], axis=1).fillna(0)

pos_columns = ['PRP','MD','JJ','JJR','JJS','RB','RBR','RBS', 'NN', 'NNS','VB', 'VBS', 'VBG','VBN','VBP','VBZ']
for pos in pos_columns:
  if pos not in emails.columns:
    emails[pos] = 0

emails = emails[['text'] + pos_columns]

emails['Adjectives'] = emails['JJ']+emails['JJR']+ emails['JJS']
emails['Adverbs'] = emails['RB']+emails['RBR'] + emails['RBS']
emails['Nouns'] = emails['NN']+emails['NNS']
emails['Verbs'] = emails['VB']+emails['VBS']+emails['VBG']+emails['VBN']+emails['VBP'] +emails['VBZ'] 

emails = emails[['text']+['PRP']+['MD']+['Adjectives']+['Adverbs']+['Nouns']+['Verbs']]


#####################################################################################################################################################################

print(emails)
