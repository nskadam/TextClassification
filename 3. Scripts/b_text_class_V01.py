# -*- coding: utf-8 -*-
"""
Created on Sat Aug 08 10:38:52 2015

@author: nilesh
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics.pairwise import linear_kernel
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier, ElasticNetCV, RidgeClassifier
from sklearn import cross_validation
import sys

# User Defined Veriables ------------------------------------------------------
path_base = 'E:/OneDrive/2. Projects/7. Perosonal/59. Text Classification 20 news'
dir_data = '1. Data'
dir_data_proc = '2. Data Processing'
fl_input = 'b_a_20news.csv'
fl_cat_key = 'a_a_category_keywords.csv'

# Create Directory Structure --------------------------------------------------
if not os.path.exists(os.path.join(path_base, dir_data_proc, 'c_a_cosine_tfidf_transformer')):
    os.makedirs(os.path.join(path_base, dir_data_proc, 'c_a_cosine_tfidf_transformer'))
if not os.path.exists(os.path.join(path_base, dir_data_proc, 'c_b_sklearn_pipline_model')):
    os.makedirs(os.path.join(path_base, dir_data_proc, 'c_b_sklearn_pipline_model'))



def train_cosine_dist_model(text, save_model = True):
    tfidf = TfidfVectorizer().fit(text)
    if save_model:
        joblib.dump(tfidf,os.path.join('c_a_cosine_tfidf_transformer', 
        'c_a_cosine_tfidf_transformer.pkl'))
    return tfidf

def score_cosine_dist_model(tfidf, cat_text, cat, text):
    cat_text_tfidf = tfidf.transform(cat_text)    
    categories = []
    for txt in text:
        # txt = text[3]
        txt_tfidf = tfidf.transform([txt])
        prob = linear_kernel(cat_text_tfidf, txt_tfidf).flatten()
        idx = prob.argmax()
        categories.append(cat[idx])
    return categories

def train_model_sklearn_pipeline(dat, save_model = True):
    
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                  'tfidf__use_idf': (True, False),}
    model = GridSearchCV(pipeline, parameters, n_jobs=1, verbose = 1)
    model = model.fit(dat.text, dat.category)
    # model = model.fit(dat.text[:1000], dat.category[:1000])

    if save_model:
        joblib.dump(model,os.path.join('c_b_sklearn_pipline_model', 
        'c_b_sklearn_pipline_model.pkl'))
    return model

def train_validate_model_sklearn_pipeline(dat):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dat.text,
                                                                         dat.category, 
                                                                         test_size=0.2, 
                                                                         random_state=0)
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                  'tfidf__use_idf': (True, False),}
    model = GridSearchCV(pipeline, parameters, n_jobs=1, verbose = 4, scoreing = matrics.accuracy)
    model = model.fit(X_train, y_train)
    
    model.best_score_
    model.best_estimator_
    
    output_train = model.predict(X_train)
    output_test = model.predict(X_test)
    
    metrics.accuracy_score(y_train, output_train)
    metrics.confusion_matrix(y_train, output_train)
    # 0.99640256207773981
    metrics.accuracy_score(y_test, output_test)
    metrics.confusion_matrix(y_test, output_test)
    # 0.91762073957099621
    
    # full data svc accu 0.99657822699515686
    return model

def train_valid_model_sklearn_pipeline_V2(dat, save_model = True):
    
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range = (1,2))),
                         ('tfidf', TfidfTransformer(use_idf = True)),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42)),])
    parameters = {'vect__ngram_range': [(1, 2)], 
                  'tfidf__use_idf': (True),
                  'clf__alpha': (1e-3, 1e-1, 10),}
    model = GridSearchCV(pipeline, parameters, n_jobs=1, verbose = 4, scoring = 'accuracy')
    model = model.fit(dat.text, dat.category)
    # model = model.fit(dat.text[:1000], dat.category[:1000])

    if save_model:
        joblib.dump(model,os.path.join('c_b_sklearn_pipline_model', 
        'c_b_sklearn_pipline_model.pkl'))
    
    dat['category_sklearn_V2'] = model.predict(dat.text)
    
    print metrics.accuracy_score(dat.category, dat.category_sklearn_V2)
    print metrics.confusion_matrix(dat.category, dat.category_sklearn_V2)
    
    best_parameters, score, _ = max(model.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

def train_valid_model_elastic_net(dat, save_model = True):
     X_train, X_test, y_train, y_test = cross_validation.train_test_split(dat.text,
                                                                         dat.category, 
                                                                         test_size=0.4, 
                                                                         random_state=0)
   
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range = (1,2))),
                         ('tfidf', TfidfTransformer(use_idf = True)),
                         ('enet',OneVsRestClassifier(RidgeClassifier())),])
    parameters = {'vect__ngram_range': [(1, 2)], 
                  'tfidf__use_idf': [True],}
    model = GridSearchCV(pipeline, parameters, n_jobs=1, cv = 10, 
                         verbose = 4, scoring = 'accuracy')
    # model = model.fit(dat.text[:1000], dat.category[:1000])
    model = model.fit(X_train, y_train)
    
    model.best_score_
    # 0.91015179433184168
    model.best_estimator_
    
    output_train = model.predict(X_train)
    output_test = model.predict(X_test)
    
    metrics.accuracy_score(y_train, output_train)
    metrics.confusion_matrix(y_train, output_train)
    # 0.99578836535930504
    metrics.accuracy_score(y_test, output_test)
    metrics.confusion_matrix(y_test, output_test)
    # 0.91314646664034738
    return 0


def validate_models(dat):
    metrics.accuracy_score(dat.category, dat.category_cosine)
    metrics.confusion_matrix(dat.category, dat.category_cosine)
    pass


def main():
    os.chdir(os.path.join(path_base, dir_data_proc))
    sys.stdout = open('log.txt', 'w')
    dat = pd.read_csv(os.path.join(path_base, dir_data_proc, fl_input), encoding = 'utf-8')
    category_keywords = pd.read_csv(os.path.join(path_base, dir_data_proc, fl_cat_key), encoding = 'utf8')
    
    # Train Model
    tfidf = train_cosine_dist_model(category_keywords.keywords.append(dat.text))
    
    # Scoring
    tfidf = joblib.load(os.path.join('c_a_cosine_tfidf_transformer', 
                                     'c_a_cosine_tfidf_transformer.pkl'))
    dat['category_cosine'] =  score_cosine_dist_model(tfidf, cat_text = category_keywords.keywords, 
                                                        cat = category_keywords.category,
                                                        text = dat.text)
    # Validation
    validate_models(dat)
    
# main()




