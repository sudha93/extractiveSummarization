import os 
import codecs
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32"
import numpy as np
import skipthoughts 
import gensim 
from gensim.models.doc2vec import TaggedDocument 
import string 
from string import maketrans
import re
from scipy import spatial 
from nltk import sent_tokenize
import time 
from sklearn.metrics import accuracy_score 
from sklearn.cluster import AgglomerativeClustering
import nltk 
from math import log10
import pickle
from heapq import nlargest
import copy 
# This method uses skip-thoughts for generating sentence representation 
# these are clustered and then we pick 'n' sentences from each cluster based 
# on TF-IDF 
# coreference resolution can be added for better results 

start_time = time.time()
N_OF_CLUSTERS = 6
S_PER_CLUSTER = 2
data = "/DATA1/USERS/anirudh/mindmap/skip-thoughts/article.txt"
#print os.listdir("/DATA1/USERS/anirudh/mindmap/skip-thoughts")

# returns a dictionary with key as cluster id and value as list of sentences in that cluster
def grouping(clusters,n,sentList):
    dictionary1 = {}
    dictionary2 = {}
    for i in range(n):
        dictionary1[i] = []
        dictionary2[i] = []
    for j,item in enumerate(clusters):
        dictionary1[item].append( sentList[j] )
        dictionary2[item].append(j)
    return dictionary1, dictionary2
            

def tfidf(clusterDict):
    n = len(clusterDict)
    # create a list of size n 
    tfList = [None] * 6
    for key in clusterDict:
        tf = {}
        sentlist = clusterDict[key]
        # tokenizing the sentlist to list of lists form
        for i,item in enumerate(sentlist):
            tokens = nltk.word_tokenize(item)
            sentlist[i] = tokens 
            
        # calculating the total no.of words in the cluster and the frequency of each word
        totalwords = 0
        for inlist in sentlist:
            totalwords = totalwords + len(inlist)
            for word in inlist:
                if word not in tf:
                    tf[word] = 1
                else:
                    tf[word] += 1
        # calculating the tf for all unique words 
        for item in tf:
            tf[item] = tf[item]/float(totalwords)
        # changing the key's value from list of sentences to list of lists of tokens
        clusterDict[key] = sentlist 
        # adding this dictionary to a list 
        tfList[key] = tf
    # calculating the no.of clusters/docs in which the word occurs 
    #print tfList 
    idf = {}
    docCount = len(tfList)
    #print docCount 
    for dictionary in tfList:
        for key in dictionary:
            if key not in idf:
                idf[key] = 1 
            else:
                idf[key] += 1
    # calculating the idf 
    for key in idf:
        idf[key] = log10(float(docCount)/idf[key])
    # calculating the tf-idf scores for each sentence
    for key in clusterDict:
        tfdict = tfList[key]
        for i,tokenlist in enumerate(clusterDict[key]):
            value = 0
            length = len(tokenlist)
            for token in tokenlist:
                tfvalue = tfdict[token] 
                idfvalue = idf[token]
                tfidf = tfvalue*idfvalue
                value = value + tfidf
                value = float(value)/length 
            clusterDict[key][i] = value 
    # return a dictionary with cluster id as key and list of sentence tfidf scores 
    return clusterDict
    





sentList = [] 
with codecs.open(data,'r') as f:
    # takes the whole doc as a string 
    total = f.read().replace('\n','')
    # split the sentences when you see exclamation,fullstop and question mark
    sentList = re.sub("([!.?])", "\\1\n", total).split('\n')
    sentList = [sent.strip() for sent in sentList if sent!='']
    # we are retaining the punctuations for the sentences
    # remove all the empty strings 
    sentList[:] = [x for x in sentList if x != '']

    # print sentList 
'''
# loading the model 
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
# encoding 
vectors = encoder.encode(sentList)
print 'finished'
print("--- %s seconds ---" % round((time.time() - start_time)))


# changes
pickle.dump(vectors, open('vectors.pkl','w'))
'''

vectors = pickle.load(open('vectors.pkl'))


#clustering 
for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=N_OF_CLUSTERS)
    clustering.fit(vectors)
    clusters = clustering.labels_
    if linkage == 'ward':
        ward_cluster,ward1 = grouping(clusters,N_OF_CLUSTERS,sentList)
    if linkage == 'average':
        avg_cluster,avg1 = grouping(clusters,N_OF_CLUSTERS,sentList)
    if linkage == 'complete':
        cmplt_cluster,cmplt1 = grouping(clusters,N_OF_CLUSTERS,sentList)
    print clustering.labels_,'\n'

# storing the sentence clusters , as it gets modified after sending it to tfidf function
print ward_cluster  #replace 
sentcluster = copy.deepcopy(ward_cluster) #replace 
finaldict = tfidf(sentcluster)
#print ward_cluster 
#print finaldict 

#finalsent = []
idlist = []
for key in finaldict:
    l = finaldict[key]
    print 'cluster',key
    maxscores= nlargest(S_PER_CLUSTER, l)
    for i in range(S_PER_CLUSTER):
        indx = l.index(maxscores[i])
        sent = ward_cluster[key][indx]   #replace
        sentId = ward1[key][indx]       #replace
        idlist.append(sentId)
        print sent 
    print '\n'
idlist.sort()
for item in idlist:
    print sentList[item]





print '---done---'
print("--- %s seconds ---" % round((time.time() - start_time)))

 

















