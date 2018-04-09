import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.stats
import seaborn as sns
#from utils.dataset import DataSet
#import utils.generate_test_splits
#dataset = DataSet()
#stance = dataset.stances
#bodies = dataset.articles
def string_to_vector(lex, text):
    words = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    features = np.zeros(len(lex))
    for word in words:
        if word in lex:
            features[lex.index(word)] += 1
    return features
################ question 1, split the dataset into a train and a validation wit ration 9:1 #######
#train_bodies_df = pd.read_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\fnc1\\fnc-1\\train_bodies.csv")
#train_stances_df = pd.read_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\fnc1\\fnc-1\\train_stances.csv")
###
#train = train_stances_df.copy()
#train['articleBody'] = 0
#train = pd.merge(train_stances_df,train_bodies_df,how = 'left')
#
#stance_type = list(train_stances_df.Stance.unique())
#Unrelated_all = len(train_stances_df[train_stances_df.Stance == stance_type[0]])/len(train_stances_df)
#Agree_all = len(train_stances_df[train_stances_df.Stance == stance_type[1]])/len(train_stances_df)
#Disagree_all = len(train_stances_df[train_stances_df.Stance == stance_type[2]])/len(train_stances_df)
#Discuss_all = len(train_stances_df[train_stances_df.Stance == stance_type[3]])/len(train_stances_df)
##
##
##
#train_subset = train.sample(frac = 0.9)
#validation_subset = train.drop(list(train_subset.index))
##
#Unrelated_train = len(train_subset[train_subset.Stance == stance_type[0]])/len(train_subset)
#Agree_train = len(train_subset[train_subset.Stance == stance_type[1]])/len(train_subset)
#Disagree_train = len(train_subset[train_subset.Stance == stance_type[2]])/len(train_subset)
#Discuss_train = len(train_subset[train_subset.Stance == stance_type[3]])/len(train_subset)
#
#Unrelated_val = len(validation_subset[validation_subset.Stance == stance_type[0]])/len(validation_subset)
#Agree_val = len(validation_subset[validation_subset.Stance == stance_type[1]])/len(validation_subset)
#Disagree_val = len(validation_subset[validation_subset.Stance == stance_type[2]])/len(validation_subset)
#Discuss_val = len(validation_subset[validation_subset.Stance == stance_type[3]])/len(validation_subset)


############ question 2, extract vector representation form all titles and bodies ############

def create_lexicon(train):
    lex = []
    for i in range(len(train)):
        a = word_tokenize( train.Headline[i].lower())
        lex += a
    for j in range(len(train_bodies_df)):
        a = word_tokenize(train_bodies_df.articleBody[j].lower())
        lex +=a
    lex1 = []
    for i in range(len(lex)):
        if lex[i][0] == "'":
            lex1.append(lex[i][1:])
        else:
            lex1.append(lex[i][:])
    #
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]
    
    word_count = Counter(lex)
    lex1= []
    for word in word_count:
            if word_count[word] < 4000 and word_count[word] > 5:
                lex1.append(word)
    return lex1

#lex_train = create_lexicon(train) 

#temp= train[['Headline','articleBody','Stance']]
#train_vector = temp.copy()
#
#lex_headline = []
#lex_body = []
#for i in range(len(train)):
#    a = word_tokenize( train.Headline[i].lower())
#    lex_headline += a
#for j in range(len(train_bodies_df)):
#    a = word_tokenize(train_bodies_df.articleBody[j].lower())
#    lex_body +=a

#body = ' '
#for i in range(len(lex_body)):
#    body = body + lex_body[i] + ' '
#    print(i)
#
#healine = ' '
#for i in range(len(lex_headline)):
#    body = body + lex_headline[i] + ' '
#    print(i)   
#train_bodies_df = pd.read_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\fnc1\\fnc-1\\competition_test_bodies.csv")
#train_stances_df = pd.read_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\fnc1\\fnc-1\\competition_test_stances.csv")
#for i in range(len(train_stances_df)):
#    headline = string_to_vector(lex,train_stances_df.Headline[i])
#    train_stances_df.Headline[i] = headline    
#    print('stance',i)
#
#for i in range(len(train_bodies_df)):
#    body = string_to_vector(lex,train_bodies_df.articleBody[i])
#    train_bodies_df.articleBody[i] = body
#    print('body',i)

#train_vector.to_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\train_vector.csv",index = False)

#A = pd.read_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\train_vector.csv")


#train_bodies_df.articleBody = string_to_vector(lex,train_bodies_df.articleBody)

def cos_sim(a,b):
    result = np.sum(a * b)/((np.linalg.norm(a) * len(a)) * (np.linalg.norm(b) * len(b)))
    return result
#a = np.zeros(len(bodies[0]))
#for i in bodies:
#    a = a + i
#b = np.zeros(len(headlines[0]))
#for i in headlines:
#    b= b+i
#cos = cos_sim(a,b)


#train = train_stances_df.copy()
#train['articleBody'] = 0
#train = pd.merge(train_stances_df,train_bodies_df,how = 'left')
#train_subset = train.sample(frac = 0.9)
#validation_subset = train.drop(list(train_subset.index))
#ts = train_subset[['Headline','articleBody']]
#vs = validation_subset[['Headline','articleBody']]
#tt = train[['Headline','articleBody']]
#
#Train_sub = np.array(ts)
#Val_sub = np.array(vs)
#Train = np.array(tt)



#a = Train
#Cos_similarity = np.zeros(len(a))
#for i in range(len(a)):
#    Cos_similarity[i]  = cos_sim(a[i][0],a[i][1]) 
##
#Cos = pd.DataFrame(Cos_similarity,columns = ['cos_sim'])
#Cos.to_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\cos_test.csv",index = False)
    
#Word_overlap = np.zeros(len(a))
#for i in range(len(a)):
#    for j in range(len(a[0][0])):
#        if a[i][0][j] > 0 and a[i][1][j] > 0:
#            Word_overlap[i] =  Word_overlap[i] + 1
#Word = pd.DataFrame(Word_overlap, columns = ['word_overlap'])
#Word.to_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\word_test.csv",index = False)

#L1 = np.zeros(len(a))
#for i in range(len(a)):
#    L1[i] = np.sum(np.abs(a[i][0]-a[i][1]))
# 
#a1 = pd.DataFrame(L1,columns = ['l1'])
#a1.to_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\l1_test.csv",index = False)
#L2= np.zeros(len(a))
#for i in range(len(a)):
#    L2[i] = np.linalg.norm(a[i][0]-a[i][1])
#a2 = pd.DataFrame(L2,columns = ['l2'])
#a2.to_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\l2_test.csv",index = False)

#kl_d = np.zeros(len(a))
#for i in range(len(a)):
#    kl_d[i] = scipy.stats.entropy(a[i][0] + (1/10000), a[i][1] +(1/10000))
#kl = pd.DataFrame(kl_d,columns = ['kl'])
#kl.to_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\kl_test.csv",index = False)


#a = np.zeros(7984)
#for i in range(44975):
#    a = a + t[i][0]
#b = np.zeros(7984)
#for i in range(44975):
#    b= b+t[i][1]
#cos = cos_sim(a,b)

############### KL-divergence ########
#kl_d = np.zeros(len(total))
#for i in range(len(total)):
#    kl_d[i] = scipy.stats.entropy(total[i][0] + (1/10000), total[i][1] +(1/10000))

#kl = pd.DataFrame(kl_d,columns = ['kl'])
#kl.to_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\kl.csv",index = False)
#    
#k = pd.read_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\kl.csv")


#z = pd.DataFrame(train_stances_df['Stance'],columns = ['Stance'])
#z['kl'] = 0
#z['word_overlap'] = 0
#z['cos_sim'] = 0
#z['l1'] = 0
#z['l2'] = 0
#
#z.kl = kl.kl
#z.word_overlap = Word.word_overlap
#z.cos_sim = Cos.cos_sim
#z.l1 = a1.l1
#z.l2 = a2.l2

#z.to_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\test_input.csv",index = False)

#zz = pd.read_csv("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\test_input.csv")
#zz = pd.concat([zz,pd.get_dummies(zz.Stance,prefix='stance')],axis=1)
#zz = zz.drop('Stance',axis=1)

#sns.distplot(z.l2[z.Stance == 'unrelated'])
    
#z_input = zz[['kl','word_overlap','cos_sim','l1','l2']]
#
#z_output = zz[['stance_agree','stance_disagree','stance_discuss','stance_unrelated']]
##
#for i in list(z_input.columns):
#    z_input[i] = (z_input[i] - z_input[i].min())/(z_input[i].max() - z_input[i].min())
#
#z_input = np.array(z_input)
#z_output = np.array(z_output)

#w = np.zeros((4,5))

#z_array = np.load("G:\\Business Analytics\\Information Retrival and Data Mining\\code for ir\\model_training_output.npy")







