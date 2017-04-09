import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



# this unction takes each list of ingredients and
# converts it into a comma separated string 
# then it uses that string of ingredients and constructs tfidf matrix from that
def to_tfidf_matrix(ingredient_list) :
	simplified_list = [','.join(words).strip() for words in ingredient_list]
	vectorizer = TfidfVectorizer(stop_words='english')
	tfidf_matrix = vectorizer.fit_transform(simplified_list).todense()
	return tfidf_matrix


def X_Y_split(data) :
	X = data['ingredients']
	Y = data['cuisine']
	return X, Y