from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
from datetime import datetime
import os
import pandas as pd
import numpy as np

t0 = datetime.strptime('2014-01-01', "%Y-%m-%d")

def pre_processingData(X):
    '''
    Used for data preprocessing: adding, deleting and transforming features.
    Args: 
        data
    Output:
        transformed data
    '''
    X=X.copy()
    X['adcreated'] = pd.to_datetime(X['adcreated'])
    X['antype'] = X['apartmentnumber'].fillna(0).astype(str).apply( lambda x: 0*(x[0]=='H')+1*(x[0]=='L')+2*(x[0]=='U')+3*(x[0]=='K'))
    delta = X['adcreated'] - t0
    X.insert(0, 'days', delta.dt.days)
    S_age = X['adcreated'].apply(lambda x: x.year) - X['buildyear']
    X.insert(0, 'age', S_age)
    X=X.drop(['apartmentnumber','adcreated'],axis=1)
    X.fillna(X.mean(), inplace=True)
    return X

class PreprocessData(BaseEstimator):
    '''
    Custom transformer for data preprocessing for pipeline
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pre_processingData(X)
    
class Model(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
       # self.model1=model1
       # self.model2=model2
        return self

    def predict(self, X,model1=None, model2=None):
        self.model1=model1
        self.model2=model2
        prediction1=self.model1.predict(X)
        prediction2=self.model2.predict(X)
        preds=[]
        for i in range(0,len(list(prediction2))):
            preds.append(list(prediction2)[i][0])
        return [prediction1,np.array(preds).astype(np.float64),(prediction1+np.array(preds).astype(np.float64))/2]

def coords2distance(X0,X1) :
    # X0: scalars [lng,lat].
    # X1: vectors [lng,lat], 
    R = 6373.0    # radius of the Earth

    lng0 = np.radians(X0[0])
    lat0 = np.radians(X0[1])
    lng1 = np.radians(X1[:,0])
    lat1 = np.radians(X1[:,1])
    
    dlng = lng1 - lng0
    dlat = lat1 - lat0
    a = np.sin(dlat/2)**2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlng/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c *1000  # Distance in meters
    return distance
   
def create_location_points(df,dist_m) :
    ind = df.reset_index().index.to_list()
    X = df[['lng','lat']].to_numpy()
    lng_l = []
    lat_l = []


    while len(ind) > 0 :
        # Pick location index and find neaby locations
        ii = ind.pop()
        i_d = np.where(coords2distance(X[ii],X[ind] ) < dist_m )[0]
        i_d = [ind[i] for i in i_d ]
        
        # Remove nearby locations and add average to output list
        Ni = len(i_d)
        lng_l.append( (np.sum(X[i_d,0])+X[ii,0])/(Ni+1) )
        lat_l.append( (np.sum(X[i_d,1])+X[ii,1])/(Ni+1) )
        for i in i_d :
            ind.remove(i)

    return lng_l, lat_l


def distance_feature_transform(dist):
    return np.maximum(2-dist,0)


def add_geofeatures(df, df_ds, dist_m):

    df_geo = df.copy()
    features_geo = list(df.columns)
    for i in range( len(df_ds) ):
        feature_name = 'dist'+str(i)
        
        dist = coords2distance(df_ds.iloc[i].values, df[['lng','lat']].values)
        df_geo = df_geo.copy()
        df_geo[feature_name] = distance_feature_transform(dist/dist_m)

        features_geo.append(feature_name)
    
    return df_geo, features_geo

def CreateLocationPoints(df_sel, dist_m):
    
    df_ds = df_sel[['lng','lat']]
    for i in range(5) :
        lng_l, lat_l = create_location_points(df_ds,dist_m)
        df_ds = pd.DataFrame({'lng':lng_l, 'lat':lat_l,})
    return df_ds
    

class GeoFeatures(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, df_sel, y=None):
        dist_m = 500
        df_ds=CreateLocationPoints(df_sel,dist_m)
        self.df_ds=df_ds
        return self

    def transform(self, X):
        dist_m = 500
        X_dataset_geo, features_geo = add_geofeatures(X, self.df_ds, dist_m)
        return X_dataset_geo
