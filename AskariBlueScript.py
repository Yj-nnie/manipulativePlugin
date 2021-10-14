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

def extractGrammar(email): 
    # Updated calculate the tags I need 
    #tag_count_data = pd.DataFrame(email['POS_Tag'].map(lambda x: Counter(tag[1] for tag in x)).to_list())
    tag_count_data = Counter([x[1] for x in email['POS_Tag']])
   
    #Convert the Counter object to dict
    tag_count_dict = dict(tag_count_data)

    #Turning dict into Series
    email_tag = pd.DataFrame(pd.Series(tag_count_dict).fillna(0).rename_axis('Tag'))
    email_tag = email_tag.reset_index()

    #use set_index to set Tag column values to be column names
    email_tag= email_tag.set_index("Tag").T.reset_index(drop=True).rename_axis(None, axis=1) 
#    print('DF TAGS ----------->\n',df_tag)

#     NNP   IN   CD   DT    (    NN     :  ...  RBR  EX  RBS  WDT     PDT  UH
#0  3966  643  589  434  324  4104  3271  ...    6   4    4    5  210    1   1
    
    #select Tags that I need
    pos_columns = ['PRP','MD','JJ','JJR','JJS','RB','RBR','RBS', 'NN', 'NNS','VB', 'VBS', 'VBG','VBN','VBP','VBZ']
    for pos in pos_columns:
      if pos not in email_tag.columns:
        email_tag[pos] = 0

    email_tag = email_tag[pos_columns] 

    print("this---->\n", email_tag)


    email_tag['Adjectives'] = email_tag['JJ']+email_tag['JJR']+ email_tag['JJS']
    email_tag['Adverbs'] = email_tag['RB']+email_tag['RBR'] + email_tag['RBS']
    email_tag['Nouns'] = email_tag['NN']+email_tag['NNS']
    email_tag['Verbs'] = email_tag['VB']+email_tag['VBS']+email_tag['VBG']+email_tag['VBN']+email_tag['VBP'] +email_tag['VBZ'] 
  
    email_tag = email_tag[['PRP']+['MD']+['Adjectives']+['Adverbs']+['Nouns']+['Verbs']]
#    print("this---->\n", email_tag)
#        PRP  MD  Adjectives  Adverbs  Nouns  Verbs
#0   59  50        2080       87   4474   1118
    return email_tag


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
emails=emails.apply(extractGrammar, axis = 1)
print(emails)


