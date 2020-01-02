import os
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn import metrics 
import joblib


TEST_DATA="input/test.csv"
   

MODEL="randomforest"
FOLD=0

def predict():
    df=pd.read_csv(TEST_DATA)
    test_idx=df["id"].values
    print(test_idx)
    preds=None

    for FOLD in range(2,3):
        print(FOLD)
        df=pd.read_csv(TEST_DATA)
        encoders=joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_label_encoder.pkl"))
        columns=joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_columns.pkl"))
        model=joblib.load(os.path.join("models",f"{MODEL}_{FOLD}.pkl"))
        for c in encoders:

            df[c]  =encoders[c].fit_transform(df[c].values.tolist())
        df=df[columns]
        preds=model.predict_proba(df)[:,1]
    
    sub=pd.DataFrame(np.column_stack((np.array(test_idx).astype(np.object),preds)),columns=["id","target"])
    return(sub)        
if __name__ == "__main__":
    submission=predict()
    submission.to_csv(f"models/{MODEL}.csv",index=False)