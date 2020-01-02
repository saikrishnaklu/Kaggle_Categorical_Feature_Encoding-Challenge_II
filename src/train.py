from __future__ import print_function
import os
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.metrics import accuracy_score,roc_auc_score
import pandas as pd
import joblib
import numpy as np
#import dispatcher

folds={
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[1,0,3,4],
    3:[1,2,0,4],
    4:[1,2,3,0],
}


TRAINING="input/train_folds.csv"
TEST="input/test.csv"
FOLD=2
MODEL="randomforest"
""" TRAINING=os.getenv(['TRAINING'])
TEST=os.getenv(['TEST'])
FOLD=os.getenv(['FOLD'])
MODEL=os.getenv(['MODEL']) """
MODELS={
    "randomforest":ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,verbose=2),
    "extratreesclassfier":ensemble.ExtraTreesClassifier(n_estimators=200,n_jobs=-1,verbose=2)
}

if __name__=="__main__":
    df_train_main=pd.read_csv(TRAINING)
    df_test=pd.read_csv(TEST)
    
    df_train=df_train_main[df_train_main.kfold.isin(folds.get(FOLD))].reset_index(drop=True)
    
    df_val=df_train[df_train_main.kfold==0].reset_index(drop=True)
    

    df_y_train=df_train.target.values
    df_y_val=df_val.target.values
    
    df_train=df_train.drop(['id','target','kfold'],axis=1)
    df_val=df_val.drop(['id','target','kfold'],axis=1)
    
    df_val=df_val[df_train.columns]
    
    label_encoders={}
    for c in df_train.columns:
        label_encoder=preprocessing.LabelEncoder()
        label_encoder.fit(df_train[c].values.tolist()+df_val[c].values.tolist()+df_test[c].values.tolist())
        onehot_encoder=preprocessing.OneHotEncoder()
        df_train.loc[:,c]=label_encoder.transform(df_train[c].values.tolist())
        df_val.loc[:,c]=label_encoder.transform(df_val[c].values.tolist())
        label_encoders[c]=label_encoder  
model=MODELS[MODEL]
model.fit(df_train,df_y_train)
pred=model.predict_proba(df_val)[:,1]
pred2=model.predict(df_val)
print(roc_auc_score(df_y_val,pred))
print(accuracy_score(df_y_val,pred2))

joblib.dump(label_encoders,f"models/{MODEL}_{FOLD}_label_encoder.pkl")
joblib.dump(model,f"models/{MODEL}_{FOLD}.pkl")
joblib.dump(df_train.columns,f"models/{MODEL}_{FOLD}_columns.pkl")
