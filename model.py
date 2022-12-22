import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
df1=pd.read_csv("Bengaluru_House_Data.csv")
# print(df1.head())
# print(df1.shape)
# print(df1.groupby('area_type')['area_type'].agg('count'))
df2=df1.drop(["area_type","society","balcony","availability"],axis="columns")
# print(df2.head())
# DATA CLEANING
# print(df2.isnull().sum())
# NA is very small as compared to dataset so we can drop NA
df3=df2.dropna()
# print(df3.isnull().sum())
# print(df3.shape)
# print(df3["size"].unique())     AS BHK AND BEDROOM TELLS US THE SAME THING SO WE WILL MERGE USE NEW COLUMN AS BHK
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))  # SPLITTING THE DATA IN SIZE COLUMN AND TAKING ONLY THE FIRST PART AS USING IT IN COLUMN NAME BHK
# print(df3.head())
# print(df3['bhk'].unique()) # some bedrooms are even with 43 bedrooms lets have a look at then
# print(df3[df3.bhk>20])
# print(df3.total_sqft.unique()) # we can see some rows have a range instead of values
def is_float(x):             # A FUNCTION TO CHECK WHETHER IT IS A INTEGER OR A RANGE
    try:
        float(x)
    except:
        return False
    return True
print(df3[~df3['total_sqft'].apply(is_float)]) #returnig the rows with range
def covert_sqft_to_num(x):           # function to take every range and covert into avrg
    tokens =x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try :
        return float(x)
    except :
        return None
# print(covert_sqft_to_num('1234'))  # just checking the function
# print(covert_sqft_to_num('1234-5678'))
df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(covert_sqft_to_num)
print(df4.head())
df5=df4.copy()
df5["price_per_sqft"]=df4["price"]*100000/df4["total_sqft"]
print(df5.head())
#NOW WE WILL WORK WITH THE LOCATION COLUMN
print(len(df5.location.unique())) # a large no. of location can nogt use one hot encoding
df5.location=df5.location.apply(lambda x: x.strip()) #removing starting and ending space if any
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)
print(len(location_stats[location_stats<=10])) #out of 1304 locations 1052 have frequency less than 10 which is good
location_stats_lessthan_ten=location_stats[location_stats<=10]
print(location_stats_lessthan_ten)
df5.location=df5.location.apply(lambda x: "other" if x in location_stats_lessthan_ten else x) # giving value other to the the location of frequency less then 10
print(len(df5.location.unique())) # only 242 left so we can apply one hot encoding now
# OUTLIERS DETECTION AND REMOVAL
print(df5[df5.total_sqft/df5.bhk<300].head())
print(df5.shape)
df6=df5[~(df5.total_sqft/df5.bhk<300)]    #removing outliers
print(df6.shape)
# print(df6.price_per_sqft.describe()) # just checking the min and max
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df= subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7=remove_pps_outliers(df6)
print(df7.shape)
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location) & (df.bhk==2)]
    bhk3=df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price,marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()
    plt.show(block=True)
    print(plt.style.available)
# plot_scatter_chart(df7,"Hebbal")
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' :bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats =bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
               exclude_indices= np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8= remove_bhk_outliers(df7)
print(df8.shape)
# plot_scatter_chart(df8,"Hebbal") # all the outliers are removed (those with same sqft area with less bhk AND MORE PRICE)
# plt.hist(df8.price_per_sqft,rwidth=0.8)
# plt.xlabel("PRICE PER SQUARE FEET")
# plt.ylabel("COUNT")
# plt.show(block=True)
# print(plt.style.available)
# plt.hist(df8.bath,rwidth=0.8)
# plt.xlabel("NO OF BATHROOM")
# plt.ylabel("COUNT")
# plt.show(block=True)
# print(plt.style.available)
# LETS ROOM THE DATA WHEN THE BATH ROOMS ARE 2 MORE THEN THE BEDROOMS
df9=df8[df8.bath<df8.bhk+2]
print(df9.shape)
# THE LAST THING IS TO DROP THE UNNECCESSARY COLUMNS LIKE SIZE AND PRICE PER SQRE FEET(USED TO FIND OUTLIERS)
df10=df9.drop(['size','price_per_sqft'],axis='columns')
# print(df10.head())
dummies=pd.get_dummies(df10.location)
df11=pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
print(df11.head())
df12=df11.drop('location',axis='columns')
print(df12.shape)
# STARTING THE TRAINING PART
X=df12.drop('price',axis='columns')
y=df12.price
# STARTING WITH THE TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(X_train,y_train)
print(lr_clf.score(X_test,y_test))
# NOW USING THE KFOLD
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
print(cross_val_score(LinearRegression(),X,y,cv=cv))
# TESTING SOME MORE TECHNIQUE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
def find_best_model_using_gridSearchcv(X,y):
    algos={
        'linear_regression' :{
            'model':LinearRegression(),
            'params':{
                'normalize':[True,False]
            }
        },
        'lasso' :{
            'model':Lasso(),
            'params':{'alpha':[1,2],
                      'selection':['random','cyclic']
                      }
        },
        'decesion_tree':{
        'model': DecisionTreeRegressor(),'params':{'criterion':['mse','friedman_mse'],'splitter':['best','random']
                                                   }
        }
    }
    scores=[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs=GridSearchCV(config['model'],config['params'],cv=cv, return_train_score= False)
        gs.fit(X,y)
        scores.append({
            'model':algo_name,
            'best_score': gs.best_score_,
            'best_params':gs.best_params_
       })
    return pd.DataFrame(scores ,columns=['model','best_score','best_params'])
print(find_best_model_using_gridSearchcv(X,y))
def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index >=0:
        x[loc_index]=1
    return lr_clf.predict([x])[0]
print(predict_price('1st Phase JP Nagar',1000,2,2))
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)
import json
columns={
    'data_columns':[col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps
            (columns))

