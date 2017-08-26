
# coding: utf-8

# In[25]:

import nltk
import csv
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


# In[26]:

f=open('/home/aman/Desktop/QLearn Aman_Malali/train_ques.txt','rU')   #Reads in the training questions text file
with open('/home/aman/train_labels.csv','rb') as k:
    reader=csv.reader(k)
    train_labels=list(reader)                                         #Reads in the training labels file
train_labels.remove(train_labels[0])                                  #removes 'id' and 'label' from the label file



# In[27]:

train_data=f.read()


# In[28]:

train_sent=train_data.splitlines()  
train_sent.remove(train_sent[0])                                          #split the training set into its corresponding 
#print len(train_sent)                                                #sentences


# In[29]:

final_set=[]
all_words1=[]
token=nltk.RegexpTokenizer(r'\w+')                  #the word tokenizer that does not read in punctuation
all_words=token.tokenize(train_data)                #All words in the file are tokenized
for j in all_words:                                  
    if j.isdigit() is False:                        #Read in only non numerical words present in the entire train set
        all_words1.append(j)
e=0
for i in train_sent:                    # Creates a list of list of lists with words of each question and the 
    words=[]                            # corresponding label [0-6]
    set1=[]
    set2=[]
    words=nltk.word_tokenize(i)
    set1.append(words[2:]) 
    set1.append(train_labels[e][1])
    final_set.append(set1)
    e=e+1

    


# In[30]:

all_words2=nltk.FreqDist(all_words1)    #The frequency distribution of all of the words present in the train file
word_features=list(all_words2.keys())
#print len(word_features)


# In[31]:

def find_features(sent):                # Finding the features of each question and storing it as a dictionary
    words2=set(sent)
    features={}
    for w in word_features:
        features[w]=(w in words2)
    return features


# In[32]:

featuresets=[(find_features(rev),category) for (rev, category) in final_set]
# Finds all the features of all the questions present in the training set and puts it in the form of a list


# In[36]:

training_set=featuresets[:3000]
#testing_set=featuresets[2900:]
#Split of 80:20 for training and testing set


# In[37]:

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
#print nltk.classify.accuracy(LinearSVC_classifier, testing_set)


# In[35]:

x=open('/home/aman/test_ques.txt','rU')     #Opening the testing data and follow the same procedure as for training
test_data=x.read()                          #data
test_set=test_data.splitlines()
test_set.remove(test_set[0])


# In[19]:

final_test=[]                               #Putting all the words in the same form as that for training data
for i in test_set:
    words=[]
    set1=[]
    words=nltk.word_tokenize(i)
    set1.append(words[2:]) 
    final_test.append(set1)


# In[20]:

answer=[['Id','Prediction']]


# In[22]:

id1=3001                                   #Predicting for all the testing data and writing it in a list
for r in final_test:
    prediction=LinearSVC_classifier.classify(find_features(r[0]))
    tempset=[id1,prediction]
    id1=id1+1
    answer.append(tempset)


# In[23]:

with open("Submission.csv",'wb') as y:   #Converting the list of prediction to a .csv file
    writer=csv.writer(y)
    writer.writerows(answer)


# In[ ]:



