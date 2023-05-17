import koco
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocess

def data_setup():
  train_dev = koco.load_dataset('korean-hate-speech', mode='train_dev')

  train_comments = []
  train_hates = []

  for i in tqdm(train_dev['train'], desc = 'Train_Valid_Dataset'):
    train_comments.append(i['comments'])
    train_hates.append(i['hate'])

  train = pd.DataFrame({'comments':train_comments,
                      'hate': train_hates
                      })
  train = train.loc[(train.hate == 'none') | (train.hate == 'hate')]
  train['hate'] = train.hate.map(lambda x: 0 if x == 'none' else 1)
  train, valid = train_test_split(train, stratify = train['hate'], test_size = 0.2)
  print('\nTrain, Valid DataSet Complete')
  print('-' * 100)


  test_comments = []
  test_hates = []

  for i in tqdm(train_dev['dev'], desc = 'Test_DataSet'):
    test_comments.append(i['comments'])
    test_hates.append(i['hate'])

  test = pd.DataFrame({'comments':test_comments,
                      'hate': test_hates
                      })
  test = test.loc[(test.hate == 'none') | (test.hate == 'hate')]
  test['hate'] = test.hate.map(lambda x: 0 if x == 'none' else 1)
  train = train.reset_index(drop = True)
  valid = valid.reset_index(drop = True)

  print('\n\nTest DataSet Complete')
  return preprocess(train), preprocess(valid), preprocess(test)