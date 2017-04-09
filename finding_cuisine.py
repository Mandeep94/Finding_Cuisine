import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

def split_data(X, Y, test_size) :
	return train_test_split(X, Y, test_size=test_size, random_state=0)

def benchmark_model_efficiency(X_train, X_test, Y_train, Y_test) :
	clf = DecisionTreeClassifier(criterion='gini', min_samples_split=2, min_samples_leaf=1)
	clf.fit(X_train, Y_train)
	pred = clf.predict_proba(X_test)
	score = log_loss(Y_test, pred)
	return score

#learner : algorithm to be trained on
def train_predict(learner, X_train, Y_train, X_test, Y_test):
	results = {}

	start = time()
	learner.fit(X_train, Y_train)
	end = time()
	results['train_time'] = end - start
	
	start = time()
	pred_train = learner.predict_proba(X_train)
	pred_test = learner.predict_proba(X_test)
	end = time()
	results['pred_time'] = end - start

	results['train_score'] = log_loss(Y_train, pred_train)
	results['val_score'] = log_loss(Y_test, pred_test)

	return results


def evaluate(results):
  
    fig, ax = plt.subplots()
    index = np.arange(2)
    bar_width = 0.3
    opacity = 0.8
    plt.bar(index, (results['LogisticRegression']['train_time'], results['LogisticRegression']['pred_time']), alpha=opacity, color='b', label='LogisticRegression', width=0.3)
    plt.bar(index+bar_width, (results['RandomForestClassifier']['train_time'], results['RandomForestClassifier']['pred_time']), alpha=opacity, color='g', label='RandomForestClassifier', width=0.3)
    plt.bar(index+bar_width*2, (results['SGDClassifier']['train_time'], results['SGDClassifier']['pred_time']), alpha=opacity, color='r', label='SGDClassifier', width=0.3)
    
    plt.ylabel('Time')
    plt.title('Time by classifier')
    plt.xticks(index + bar_width, ('train_time', 'pred_time'))
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    plt.bar(index, (results['LogisticRegression']['train_score'], results['LogisticRegression']['val_score']), alpha=opacity, color='b', label='LogisticRegression', width=0.3)
    plt.bar(index+bar_width, (results['RandomForestClassifier']['train_score'], results['RandomForestClassifier']['val_score']), alpha=opacity, color='g', label='RandomForestClassifier', width=0.3)
    plt.bar(index+bar_width*2, (results['SGDClassifier']['train_score'], results['SGDClassifier']['val_score']), alpha=opacity, color='r', label='SGDClassifier', width=0.3)
    
    plt.ylabel('Time')
    plt.title('Time by classifier')
    plt.xticks(index + bar_width, ('train_score', 'val_score'))
    plt.legend()

    plt.tight_layout()
    plt.show()



class EncodeDecode:

    def __init__(self, clist):
        self.clist = clist
        self.le = LabelEncoder()
        self.le.fit(list(clist))

    def encode_labels(self, cuisine_list):
        encoded_list = self.le.transform(cuisine_list)
        return encoded_list


    def decode_labels(self, encoded_labels) :
        return self.le.inverse_transform(encoded_labels)