from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn import decomposition,pipeline
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn import ensemble, preprocessing, grid_search, cross_validation
from sklearn import metrics 
from sklearn.calibration import CalibratedClassifierCV

import scipy.stats as scs
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


import scipy as sp

import random
random.seed(12)


def part(n, k):
    def _part(n, k, pre):
        if n <= 0:
            return []
        if k == 1:
            if n <= pre:
                return [[n]]
            return []
        ret = []
        for i in range(min(pre, n), 0, -1):
            ret += [[i] + sub for sub in _part(n-i, k-1, i)]
        return ret
    return _part(n, k, n)     
            

def labels(x):
    x.values[0]
    lol  = 0 
    l = {'Benign':0 , 'LikelyPathogenic':0 ,'Pathogenic':0,'VUS_I':0}
    for i in range(0,len(x)):
        try:
            l[x[i]] = l[x[i]] + 1
            lol = lol+1
        except:
            continue
    return pd.Series({'Benign':l['Benign'],'LikelyPathogenic':l['LikelyPathogenic'],'Pathogenic':l['Pathogenic'],'VUS_I':l['VUS_I']})
    #print x[0],x[1]




def logloss(act, pred):
    #print act,pred
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

    

def tf_idf_cal(train,test,y):
    random.seed(1)
    print '0'
    #print type(x['e1'],x['e2'],x['e3'],x['e4'],x['e5'],x['e6'],x['e7'],x['e8'],x['e9'],x['e10'],x['e11'],x['l1'],x['l2'],x['l3'],x['l4'],x['l5'],x['l6'],x['l7'],x['l8'],x['l9'],x['l10'],x['l11'],x['l12'],x['l13'],x['l14'],x['l15'],x['l16'],x['l17'],x['l18'],x['l19'],x['l20'])
    traindata = list(train.apply(lambda x:'%s' % (x['VariantAllele']),axis=1))
    testdata = list(test.apply(lambda x:'%s' % (x['VariantAllele']),axis=1))
    
    # the infamous tfidf vectorizer (Do you remember this one?)
    print '1'
    tfv = TfidfVectorizer(min_df=1,  max_features=None, 
            strip_accents='unicode', analyzer='char',token_pattern=r'\w{1,}',
            ngram_range=(1, y), use_idf=False,smooth_idf=False,sublinear_tf=1,
            stop_words = ['_'])
    
    # Fit TFIDF
    tfv.fit(list(traindata)+list(testdata))
    print '1'
    X =  tfv.transform(traindata).toarray()
    X_test1 = tfv.transform(testdata).toarray()
    X[X != 0] = 1
    X_test1[X_test1 !=0] = 1
    
    return X , X_test1,tfv



def count(x):
    l = {'A':0 , 'C':0 ,'G':0,'T':0}
    x = x.values[0]
    #print x,'x'
    for i in x:
        l[i] = l[i]+1
    return pd.Series({'A':l['A'],'C':l['C'],'G':l['G'],'t':l['T']})


def count1(x):
    l = {'A':0 , 'C':0 ,'G':0,'T':0}
    x = x.values[0]
    #print x,'x'
    for i in x:
        l[i] = l[i]+1
    return pd.Series({'A1':l['A'],'C1':l['C'],'G1':l['G'],'T1':l['T']})


if __name__ == '__main__':
    
    train = pd.read_csv('../input/train.csv',header = None)
    test = pd.read_csv('../input/test.csv',header = None)

    index_train = pd.read_csv('../input/index_train.csv')
    index_test = pd.read_csv('../input/index_test.csv')    
    
    print 'zero one'
    col_train = ['id','Chromosome','Location','Reference','VariantAllele','Reference Length','VariantAllele Length','Gene','DBSNP','1000Genome','Exome server'\
    ,'Exome_consortium_Fraction','assigned_l1','assigned_l2','assigned_l3','assigned_l4','assigned_l5','assigned_l6','assigned_l7','assigned_l8','assigned_l9','assigned_l10',
    'assigned_l11','assigned_l12','assigned_l13','SPLICING_Type','Primates Conservation','Mammal Conservation','Vertebrate Conservation','Protein shortening'\
    ,'isTruncating','isMissense','isMissenseType','33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52','53',\
    '54','55','56','57','58','59','60','Damaging prediction count','Impacts gene expression','functionally validated','pathogenic variant','is novel','label']
    
    col_test = ['id','Chromosome','Location','Reference','VariantAllele','Reference Length','VariantAllele Length','Gene','DBSNP','1000Genome','Exome server'\
    ,'Exome_consortium_Fraction','assigned_l1','assigned_l2','assigned_l3','assigned_l4','assigned_l5','assigned_l6','assigned_l7','assigned_l8','assigned_l9','assigned_l10',
    'assigned_l11','assigned_l12','assigned_l13','SPLICING_Type','Primates Conservation','Mammal Conservation','Vertebrate Conservation','Protein shortening'\
    ,'isTruncating','isMissense','isMissenseType','33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52','53',\
    '54','55','56','57','58','59','60','Damaging prediction count','Impacts gene expression','functionally validated','pathogenic variant','is novel']
    
    
    
    train.columns = col_train
    
    y = train.label.values
    
    
    train  = train.drop(['label','id'],axis =1)
    test.columns = col_test
    ids = test.id.values
    test  = test.drop('id',axis =1)
    
    
    test.columns
    
    
    train_33_53 = train[['33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52','53']]
    train_labels = train[['assigned_l1','assigned_l2','assigned_l3','assigned_l4','assigned_l5','assigned_l6','assigned_l7','assigned_l8','assigned_l9','assigned_l10','assigned_l11','assigned_l12','assigned_l13']]
    train_54_60 = train[['54','55','56','57','58','59','60']]

    """
    removed_columns = []
    for i in range(1,14):
        if train.loc[pd.notnull(train['assigned_l' + str(i)])].shape[0] == 0:
            removed_columns.append('assigned_l' + str(i))
            (16,22,23) 
     """     
            
    #train = train.drop(['assigned_l1', 'assigned_l2', 'assigned_l6', 'assigned_l7', 'assigned_l10', 'assigned_l13'],axis =1)
 
    train = train.drop(['assigned_l1', 'assigned_l2', 'assigned_l6', 'assigned_l7', 'assigned_l10', 'assigned_l13'],axis =1)
    test = test.drop(['assigned_l1', 'assigned_l2', 'assigned_l6', 'assigned_l7', 'assigned_l10', 'assigned_l13'],axis =1)
    train_labels = train_labels.drop(['assigned_l1', 'assigned_l2', 'assigned_l6', 'assigned_l7', 'assigned_l10', 'assigned_l13'],axis =1)
    
    
    
    l = train['Gene'].append(test['Gene'])
    l = l.reset_index()
    l = l.Gene.value_counts().reset_index()
    
    l.columns = ['Gene','count']
    
    train = train.merge(l, on = 'Gene' , how = 'left')
    test = test.merge(l, on = 'Gene' , how = 'left')
       
    train[['Exome server','Exome_consortium_Fraction','Location','Gene']]
   
    more_f =  []
    for f in train.columns:
        if train[f].dtype=='object':
            more_f.append(f)
            #print f
            if f != 'loca':   
                #print 'assa',f
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(train[f].values) + list(test[f].values))
                train[f] = lbl.transform(list(train[f].values))
                test[f] = lbl.transform(list(test[f].values))
                
    train.replace(np.nan , -1 ,inplace = True)
    test.replace(np.nan , -1 ,inplace = True)
    k = train.columns
    
    
    print 's'
    #0.9693(cross validation score)
    gbm1 = ensemble.GradientBoostingClassifier(random_state = 1, learning_rate = 0.05 ,  min_samples_split = 2,subsample = 1 , max_features = 0.55 ,  n_estimators = 400 , max_depth = 14)               
    gbm1.fit(train,y)
    prob1 = gbm1.predict(test)
    submission = pd.DataFrame({'id':ids, 'predicted':prob1})
    submission.to_csv('final_if_other_not_done.csv',index = False)
    
    
    
