import re

def preprocess(df):
  df['comments'] = df['comments'].map(lambda x : re.sub('\([^)]*\)', '', x))
  df['comments'] = df['comments'].map(lambda x : re.sub('(\.\s?)', ' ', x))
  return df