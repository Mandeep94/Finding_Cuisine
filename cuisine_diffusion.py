import pandas as pd
import numpy as np
from pylab import *
from scipy import *
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA

# creates a dictionary with cuisines and all ingredients present in that cuisine
# includes duplicates
def create_cuisine_ingredient_dict(data):
    dict_cuisine_ing = {}
    cuisines = []
    ingredients = []

    for i in range(len(data)):
        c = data['cuisine'][i]
        ing = data['ingredients'][i]
        if c not in dict_cuisine_ing.keys():
            cuisines.append(c)
            dict_cuisine_ing[c] = ing
        else : 
            ing_list = dict_cuisine_ing[c]
            ing_list.extend(ing)
            dict_cuisine_ing[c] = ing_list
        ingredients.extend(ing)

    return dict_cuisine_ing, list(set(cuisines)), list(set(ingredients))


def count_matrix(dictionary, cuisines, ingredients):
	# we are counting number of times each ingredient occurs in a cuisine
	matrix = np.zeros((len(cuisines), len(ingredients)))
	i = 0
	for c in cuisines:
		ing = dictionary[c]
		for ingre in ing:
			# getting the index of ingreient
			j = ingredients.index(ingre)
			matrix[i][j] += 1
		i += 1

	return matrix


def tf_idf_from_count_matrix(count_matrix):
    
    countsMatrix = sparse.csr_matrix(count_matrix)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(count_matrix)
    tfidf.toarray() 
    return tfidf.toarray()


def diffusion(cuisine_ingredient_dict, cuisines, labels, reduced_data) :
	i = 0 
	j = 0 

	effect_on_cluster = [0 for cuisine in cuisines]

	# Jaccard Index = (the number in both sets) / (the number in either set) * 100
	for cuisineA in cuisines:  

	    A_intersection = 0
	    numInClusterBesidesA = 0
	    setA = set(cuisine_ingredient_dict[cuisineA])
	    setB_forA = []
	    j = 0
	    
	    for cuisineB in cuisines:
	        if cuisineB != cuisineA: # if it is A itself - we obviously wouldn't want this (will be exactly 1)
	            if labels[j] == labels[i]: #determines if then they are both in the same cluster
	                setB_forA.extend(set(cuisine_ingredient_dict[cuisineB]))
	                numInClusterBesidesA += 1
	        j += 1
	    
	    A_intersection = len(set(setA & set(setB_forA))) / float(len(set(setA.union(setB_forA))))
	    effect_on_cluster[i] = A_intersection
	       
	    i += 1

#	return effect_on_cluster


#def plot(effect_on_cluster, reduced_data, cuisines, labels):

	rdata = reduced_data
	i=0
	figureRatios = (15,20)
	x = []
	y = []
	color = []
	area = []

	#creating a color palette:
	colorPalette = ['#ff6300','#2c3e50', '#660033'] 
	# green,blue, orange, grey, purple

	plt.figure(1, figsize=figureRatios)

	for data in rdata:
	    x.append(data[0]) 
	    y.append(data[1])  
	    color.append(colorPalette[labels[i]]) 
	    area.append(effect_on_cluster[i]*27000) # magnifying the bubble's sizes (all by the same unit)
	    # plotting the name of the cuisine:
	    text(data[0], data[1], cuisines[i], size=10.6,horizontalalignment='center', fontweight = 'bold', color='w')
	    i += 1

	plt.scatter(x, y, c=color, s=area, linewidths=2, edgecolor='w', alpha=0.80) 

	plt.axis([-0.45,0.65,-0.55,0.55])
	plt.axes().set_aspect(0.8, 'box')

	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.axis('off') # removing the PC axes

	plt.show()
