from sklearn import model_selection
import pandas as pd 
df=pd.read_csv("input/train.csv")
#print(df.head())
print(df[df.target==1,["target"]].head())




for c in df_train.columns:
   label_encoder=preprocessing.LabelEncoder()
   label_encoder.fit(df_train[c].values.tolist()+df_val[c].values.tolist()+df_test[c].values.tolist())
   onehot_encoder=preprocessing.OneHotEncoder()
   df_train.loc[:,c]=label_encoder.transform(df_train[c].values.tolist())
   df_val.loc[:,c]=label_encoder.transform(df_val[c].values.tolist())
   label_encoders[c]=label_encoder