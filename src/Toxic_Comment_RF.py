from matplotlib import pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import metrics

from functions import load_bad_words, build_data_path, print_report
from constants import LABEL_COLS



BAD_WORDS = load_bad_words()
training_data_path = build_data_path('train.csv')

df = pd.read_csv(training_data_path)
X = df['comment_text']
y = df[LABEL_COLS]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)

clf = RandomForestClassifier()
tfidf = TfidfVectorizer(lowercase=True, stop_words='english')
bad_word_counter = CountVectorizer(vocabulary=BAD_WORDS)

union = make_union(tfidf, bad_word_counter)
pipeline = make_pipeline(union, clf)
optimizer = pipeline

# Auto-tune hyperparameters
while autotune_hyperparameters.lower() not in ['yes', 'no']:
    autotune_hyperparameters = input('Please enter "yes" or "no".')
if autotune_hyperparameters == 'yes':
    parameters = {
        'featureunion__tfidfvectorizer__lowercase': [True, False],
        'featureunion__tfidfvectorizer__max_features': [1000, 5000, 10000, None],
        'featureunion__countvectorizer__binary': [True, False],
        'randomforestclassifier__class_weight': [None, 'balanced'],
    }
    optimizer = GridSearchCV(pipeline, parameters, scoring='f1_weighted', verbose=3)
fit_params = {
    'randomforestclassifier__sample_weights': compute_sample_weight('balanced', y_train)
}
optimizer.fit(X_train, y_train)

y_predictions = optimizer.predict(X_valid)
print(y_predictions.shape, y_valid.shape)
# best_estimator_ = optimizer.best_estimator_

metrics.roc_auc_score(y_valid, y_predictions)

test_data = build_data_path('test.csv')
data_df = pd.read_csv(test_data)
test_labels = build_data_path('test_labels.csv')
label_df = pd.read_csv(test_labels)
test_df = data_df.set_index('id').join(label_df.set_index('id'))
CONDITIONS = [f'{label} != -1' for label in LABEL_COLS]
QUERY_STRING = ' & '.join(CONDITIONS)
test_df = test_df.query(QUERY_STRING)
X_test = test_df['comment_text']
y_test = test_df[LABEL_COLS]
y_predictions = optimizer.predict(X_test)
print_report(y_test, y_predictions, data_type='TESTING')

# The Inspiration for the Code
# has been taken from
# https://github.com/SaltyQuetzals/Toxicroak
