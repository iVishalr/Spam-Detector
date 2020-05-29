import nltk
import pandas as pd
import numpy as np

df = pd.read_table('SMSSpamCollection',header = None, encoding='UTF-8')
classes = df[0]

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

#PreProcessing using Regular Expressions

text_messages = df[1]
# use regular expressions to replace email addresses, URLs, phone numbers, other numbers
# Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
# Replace 10 digit phone numbers (formats include parenthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumber')
# Replace numbers with 'number'
processed = processed.str.replace(r'\d+(\.\d+)?', 'number')
# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')
# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')
# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')
processed = processed.str.lower()
print(processed)
#removing stop words from the text messages
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
#applying stemming algorithms
ps = nltk.PorterStemmer()
processed = processed.apply(lambda x:' '.join(ps.stem(term) for term in x.split()))
#applying tokenizing algorithms
#creating a BOW Model

from nltk.tokenize import word_tokenize

all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

print('Number of words: {}'.format(len(all_words)))

word_features = list(all_words.keys())[0:]
print(word_features)

def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

messages = zip(processed,Y)
messages = list(messages)

seed = 1
np.random.seed = seed
np.random.shuffle(messages)

featureSet = [(find_features(text),label) for (text,label) in messages]

#Creating training and testing data sets

from sklearn import model_selection

training,testing = model_selection.train_test_split(featureSet,test_size=0.20, random_state=seed)
print("Size of Training data set:",len(training))
print("Size of Testing data set:",len(testing))

#SKLearn Classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
classiType = ['Logistic Regression']
classifier = [LogisticRegression()]

models = list(zip(classiType,classifier))

from nltk.classify.scikitlearn import SklearnClassifier

for name,model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model,testing)*100
    print("Model:",name,", Accuracy",accuracy)

txt_features,labels=zip(*testing)
prediction = nltk_model.classify_many(txt_features)
print(classification_report(labels,prediction))
dataTable=pd.DataFrame(confusion_matrix(labels,prediction),
             index=[['Actual','Actual'],['Valid','Spam']],
             columns=[['Predicted','Predicted'],['Valid','Spam']])
print(dataTable)
dataTable.plot(kind="bar")
plt.show()
