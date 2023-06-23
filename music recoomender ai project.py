#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[2]:


get_ipython().system('pip install spotipy')


# In[3]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# In[6]:


spotify_df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\PROJECTS\data.csv\data.csv')
spotify_df.head()


# In[7]:


data_w_genre = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\PROJECTS\data_w_genres.csv\data_w_genres.csv')
data_w_genre.head()


# In[8]:


data_w_genre.dtypes


# In[9]:


data_w_genre['genres'].values[0]


# In[10]:


data_w_genre['genres'].values[0][0]


# In[11]:


data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
data_w_genre['genres_upd'].values[0][0]


# In[12]:


spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))


# In[13]:


spotify_df['artists'].values[0]


# In[14]:


spotify_df['artists_upd_v1'].values[0][0]


# In[15]:


spotify_df[spotify_df['artists_upd_v1'].apply(lambda x: not x)].head(5)


# In[16]:


spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )
#need to create my own song identifier because there are duplicates of the same song with different ids. I see different
spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
spotify_df.sort_values(['artists_song','release_date'], ascending = False, inplace = True)
spotify_df[spotify_df['name']=='Adore You']


# In[17]:


spotify_df.drop_duplicates('artists_song',inplace = True)
spotify_df[spotify_df['name']=='Adore You']


# In[18]:


artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')


# In[19]:


artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]


# In[20]:


artists_exploded_enriched_nonnull[artists_exploded_enriched_nonnull['id'] =='6KuQTIu1KoTTkLXKrwlLPV']


# In[21]:


artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))
artists_genres_consolidated.head()


# In[22]:


spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')


# In[23]:


spotify_df.tail()


# In[24]:


spotify_df.head()


# In[25]:


spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])


# In[26]:


float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
ohe_cols = 'popularity'
spotify_df['popularity'].describe()


# In[27]:


spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))
# tfidf can't handle nulls so fill any null values with an empty list
spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])
spotify_df.head()


# In[28]:


#simple function to create OHE features
#this gets passed later on
def ohe_prep(df, column, new_name): 
    """ 
    Create One Hot Encoded features of a specific column

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        column (str): Column to be processed
        new_name (str): new column name to be used
        
    Returns: 
        tf_df: One hot encoded features 
    """
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df


# In[29]:


#function to build entire feature set
def create_feature_set(df, float_cols):
    """ 
    Process spotify df to create a final set of features that will be used to generate recommendations

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        float_cols (list(str)): List of float columns that will be scaled 
        
    Returns: 
        final: final set of features 
    """
    
    #tfidf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
    genre_df.reset_index(drop = True, inplace=True)

    #explicity_ohe = ohe_prep(df, 'explicit','exp')    
    year_ohe = ohe_prep(df, 'year','year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_red','pop') * 0.15

    #scale float columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    #concanenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
     
    #add song id
    final['id']=df['id'].values
    
    return final


# In[30]:


complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)#.mean(axis = 0)


# In[31]:


complete_feature_set.head()


# In[32]:


client_id = '5fbb233f93b244219e40d5977626c4ae'
client_secret= '55fe6eb0802149a5b4876ff275bfa1e2'


# In[33]:


scope = 'user-library-read'

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage: %s username" % (sys.argv[0],))
    sys.exit()


# In[35]:


auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)


# In[36]:


token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='http://localhost:8888/')


# In[ ]:




