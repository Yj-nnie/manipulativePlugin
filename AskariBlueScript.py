import pandas as pd

#emailtext = pd.read_csv('m1.txt', names = ['text'])

emailtext = pd.read_csv('m1.txt', sep='\n', header=None)[0].str.cat()
df = pd.DataFrame([emailtext], columns=['text'])

print(df)