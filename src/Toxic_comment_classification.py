#Please execute the following code on the interactive python notebook and in chunks.
# The code has been compiled and put in one file for easy access


import pandas as pd
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
from statistics import mean
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

from sklearn.metrics import roc_auc_score, confusion_matrix
import statistics
from sklearn.metrics import recall_score

from wordcloud import WordCloud
from collections import Counter

from sklearn.pipeline import Pipeline

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import metrics

from functions import load_bad_words, build_data_path, print_report
from constants import LABEL_COLS

warnings.filterwarnings('ignore')

%matplotlib inline

#tryin to get the visual idea of the code base and the type of data types present in the data base

print (df.shape)
print (df.dtypes)

# now this is a feature which is called as reindexing and this seems to be very cool.
#reindex() function shuffles up the entire dataset and makes the alignment random. It can make shuffling for rows random as well as columns random
# for the reindexing along the columns you can simply use print(df1.reindex(colum, axis='columns'))
df = df.reindex(np.random.permutation(df.index))

##creating a seperate dataframe for the label field, here we put the comments in the Matrix

import numpy as np

comment = df['comment_text']
print(comment.head())

#comment = comment.as_matrix()

#a = numpy.asarray(comment)
comment = comment.to_numpy()

#creating a seperate dataframe for the label field
label = df[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
print(label.head())
label = label.to_numpy()

ct1,ct2 = 0,0
for i in range(label.shape[0]):
    ct = np.count_nonzero(label[i])
    if ct :
        ct1 = ct1+1
    if ct>1 :
        ct2 = ct2+1
print(ct1)
print(ct2)

x = [len(comment[i]) for i in range(comment.shape[0])]

print('average length of the comments present in dataset is  comment: {:.3f}'.format(sum(x)/len(x)) )
bins = [1,200,400,600,800,1000,1200]
plt.hist(x, bins=bins, color='brown')
plt.xlabel('Lengths of different comments in the dataset')
plt.ylabel('Total Number of comments in the Dataset')       
plt.axis([0, 1200, 0, 90000])
plt.grid(True)
plt.show()


#here we are trying to classify the sentences as Toxic, Severe Toxic. etc  depending on the length of the statements

y = np.zeros(label.shape)
for ix in range(comment.shape[0]):
    l = len(comment[ix])
    if label[ix][0] :
        y[ix][0] = l
    if label[ix][1] :
        y[ix][1] = l
    if label[ix][2] :
        y[ix][2] = l
    if label[ix][3] :
        y[ix][3] = l
    if label[ix][4] :
        y[ix][4] = l
    if label[ix][5] :
        y[ix][5] = l

labelsplt = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
color = ['brown','black','green','blue','grey','chartreuse']        
plt.hist(y,bins = bins,label = labelsplt,color = color)
plt.axis([0, 1200, 0, 8000])
plt.xlabel('Length of comments in the Dataset')
plt.ylabel('Number of comments in the Dataset') 
plt.legend()
plt.grid(True)
plt.show()



# Subsetting labels from the training data
train_labels = df[['toxic', 'severe_toxic',
                     'obscene', 'threat', 'insult', 'identity_hate']]
label_count = train_labels.sum()

label_count.plot(kind='bar', title='Labels Frequency', color='steelblue')



#removing comments with excessive lengths

comments = []
labels = []

for ix in range(comment.shape[0]):
    if len(comment[ix])<=400:
        comments.append(comment[ix])
        labels.append(label[ix])
      

labels = np.asarray(labels)
print(len(comments))


#we are preparing strings containing all punctuations to be removed
import string
print(string.punctuation)
punctuation_edit = string.punctuation.replace('\'','') +"0123456789"
print (punctuation_edit)
outtab = "                                         "
trantab = str.maketrans(punctuation_edit, outtab)


#updating the list of Stop words

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# Initialize the stopwords
stoplist = stopwords.words('english')

print(stoplist)

stoplist.append('')

for x in range(ord('b'), ord('z')+1):
    stoplist.append(chr(x))

stop_words=stoplist



#Stemming and lemmmatizing

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

#create objects for stemmer and lemmatizer
lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()
#download words from wordnet library
nltk.download('wordnet')


#now here we will loop through all the comments and apply punctuation removal,splitting the words by space,applying stemmer and lemmatizer, recombining the words again for further processing

for i in range(len(comments)):
    comments[i] = comments[i].lower().translate(trantab)
    l = []
    for word in comments[i].split():
        l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))
    comments[i] = " ".join(l)



#Applying Count Vectorizer - here we convert the comments into matrix of token counts which will signify the number of times it occurs.

#import required library

from sklearn.feature_extraction.text import CountVectorizer

#create object supplying our custom stop words
count_vector = CountVectorizer(stop_words)
#fitting it to converts comments into bag of words format
tf = count_vector.fit_transform(comments)
tf.toarray()
# print(count_vector.get_feature_names())
#print(tf.shape)
#print(tf)

#splitting the data set into Training and Testing 


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_y = pd.read_csv("data/test_labels.csv")



def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion)
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = shuffle(tf, labels,3)

print(X_test.shape)
print(X_train.shape)


#tokenizing


def tokenize(text):
    '''
    Tokenize text and return a non-unique list of tokenized words found in the text. 
    Normalize to lowercase, strip punctuation, remove stop words, filter non-ascii characters.
    Lemmatize the words and lastly drop words of length < 3.
    '''
    text = text.lower()
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    words = nopunct.split(' ')
    # remove any non ascii
    words = [word.encode('ascii', 'ignore').decode('ascii') for word in words]
    lmtzr = WordNetLemmatizer()
    words = [lmtzr.lemmatize(w) for w in words]
    words = [w for w in words if len(w) > 2]
    return words

#TF-IDF Vectorizer

 vector = TfidfVectorizer(ngram_range=(1, 1), analyzer='word',
                         tokenizer=tokenize, stop_words='english',
                         strip_accents='unicode', use_idf=1, min_df=10)
 X_train = vector.fit_transform(train['comment_text'])
 X_test = vector.transform(test['comment_text'])


 vector.get_feature_names()[0:20]


# Creating classifiers with default parameters initially.
clf1 = MultinomialNB()
clf2 = LogisticRegression()

def cross_validation_score(classifier, X_train, y_train):
    '''
    Iterate though each label and return the cross validation F1 and Recall score 
    '''
    methods = []
    name = classifier.__class__.__name__.split('.')[-1]

    for label in test_labels:
        recall = cross_val_score(
            classifier, X_train, y_train[label], cv=10, scoring='recall')
        f1 = cross_val_score(classifier, X_train,
                             y_train[label], cv=10, scoring='f1')
        methods.append([name, label, recall.mean(), f1.mean()])

    return methods


 # Calculating the cross validation F1 and Recall score for our 3 baseline models.
methods1_cv = pd.DataFrame(cross_validation_score(clf1, X_train, train))
methods2_cv = pd.DataFrame(cross_validation_score(clf2, X_train, train))



# Creating a dataframe to show summary of results for the training dataset.
methods_cv = pd.concat([methods1_cv, methods2_cv, methods3_cv])
methods_cv.columns = ['Model', 'Label', 'Recall', 'F1']
meth_cv = methods_cv.reset_index()
meth_cv[['Model', 'Label', 'Recall', 'F1']]


def score(classifier, X_train, y_train, X_test, y_test):
    """
    Calculate F1, Recall for each label on test dataset.
    """
    methods = []
    name = classifier.__class__.__name__.split('.')[-1]
    predict_df = pd.DataFrame()
    predict_df['id'] = test_y['id']

    for label in test_labels:
        classifier.fit(X_train, y_train[label])
        predicted = classifier.predict(X_test)

        predict_df[label] = predicted

        recall = recall_score(y_test[y_test[label] != -1][label],
                              predicted[y_test[label] != -1],
                              average="weighted")
        f1 = f1_score(y_test[y_test[label] != -1][label],
                      predicted[y_test[label] != -1],
                      average="weighted")
        methods.append([name, label, recall, f1])

    return methods

# Calculating the F1 and Recall score for our 3 baseline models.
methods1 = score(clf1, X_train, train, X_test, test_y)
methods2 = score(clf2, X_train, train, X_test, test_y)
methods3 = score(clf3, X_train, train, X_test, test_y)

# Creating a dataframe to show summary of results.
methods1 = pd.DataFrame(methods1)
methods2 = pd.DataFrame(methods2)
methods3 = pd.DataFrame(methods3)
methods = pd.concat([methods1, methods2, methods3])
methods.columns = ['Model', 'Label', 'Recall', 'F1']
meth = methods.reset_index()
meth[['Model', 'Label', 'Recall', 'F1']]


#Implementing the Adaboost Algorithm( Ensemble Method)

ab_clf = AdaBoostClassifier()
boosting_models = [ab_clf]

boosting_score_df = []
for model in boosting_models:
    f1_values = []
    recall_values = []
    training_time = []
    predict_df = pd.DataFrame()
    predict_df['id'] = test_y['id']

    for idx, label in enumerate(test_labels):
        start = timer()
        model.fit(X_train, train[label])
        predicted = model.predict(X_test)
        training_time.append(timer() - start)
        predict_df[label] = predicted
        f1_values.append(f1_score(test_y[test_y[label] != -1][label],
                                  predicted[test_y[label] != -1],
                                  average="weighted"))
        recall_values.append(recall_score(test_y[test_y[label] != -1][label],
                                          predicted[test_y[label] != -1],
                                          average="weighted"))
        name = model.__class__.__name__

    val = [name, mean(f1_values), mean(recall_values)]

    boosting_score_df.append(val)
    boosting_score = pd.DataFrame(boosting_score_df,)
    boosting_score.columns = ['Model', 'F1','Recall']
    boosting_score

 # Voting Classifier which is an ensemble of Logistic Regression and Adaboost Classifier

 	ensemble_clf = VotingClassifier(estimators=[('lr', lr_clf),
                                            ('ab', ab_clf)], voting='hard')
	ensemble_score_df = []
	f1_values = []
	recall_values = []
	predict_df = pd.DataFrame()
	predict_df['id'] = test_y['id']
	for label in test_labels:
    	start = timer()
    	ensemble_clf.fit(X_train, train[label])
    	training_time.append(timer() - start)
    	predicted = ensemble_clf.predict(X_test)
    	predict_df[label] = predicted
    	f1_values.append(f1_score(test_y[test_y[label] != -1][label],
                              	predicted[test_y[label] != -1],
                              	average="weighted"))
    	recall_values.append(recall_score(test_y[test_y[label] != -1][label],
                                      	predicted[test_y[label] != -1],
                                      	average="weighted"))
   	     name = 'Proposed_Ensemble_method'

	     val = [name, mean(f1_values), mean(recall_values),]
	     ensemble_score_df.append(val)

# printing the values

ensemble_score = pd.DataFrame(ensemble_score_df,)
ensemble_score.columns = ['Model', 'F1','Recall']
ensemble_score

# The Inspiration for the Code
# has been taken from
# https://github.com/tianqwang/Toxic-Comment-Classification-Challenge
# https://github.com/nupurbaghel/Capstone_Project_ML





