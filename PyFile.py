#!/usr/bin/env python
# coding: utf-8

# ### Objective of this project:<br>
# Recommend movies to a user-<br>
# Through similarities between the movies they've seen<br>
# and find similarity between users and recommend movies based on what the similar users like.<br>
# Using Item based filtering and Collaborative filtering  

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## About the data sets:<br>
# ### •movies.csv<br>
# contains the title, genre and movie id, the movie id being unique<br>
# the following genres exist:<br>
# -Action<br>
# -Adventure<br>
# -Animation<br>
# -Children's<br>
# -Comedy<br>
# -Crime<br>
# -Documentary<br>
# -Drama<br>
# -Fantasy<br>
# -Film-Noir<br>
# -Horror<br>
# -Musical<br>
# -Mystery<br>
# -Romance<br>
# -Sci-Fi<br>
# -Thriller<br>
# -War<br>
# -Western<br>
# -(no genres listed)<br>
# ### •ratings.csv<br>
# contains userId and the rating they've given the movie<br>
# The rating ranges from 0 to 5<br>
# ### •tags.csv<br>
# contains the tag a user has given a movie<br>
# ### •links.csv<br>
# contains the number to put in the url to find the movie on<br>
# https://movielens.org <br>
# http://www.imdb.com <br>
# https://www.themoviedb.org

# In[2]:


df_movies = pd.read_csv('movies.csv')
df_movies.head()


# In[3]:


df_links = pd.read_csv('links.csv')
df_links.head()


# In[4]:


df_ratings = pd.read_csv('ratings.csv')
df_ratings.head()


# In[5]:


df_tags = pd.read_csv('tags.csv')
df_tags.head()


# ## EDA

# In[6]:


movie_profile = df_movies
movie_profile = movie_profile.drop(columns=['title', 'genres']).set_index('movieId')
movie_profile.sort_index(axis=0, inplace=True)
user_x_movie = pd.pivot_table(df_ratings, values='rating', index=['movieId'], columns = ['userId'])
user_x_movie.sort_index(axis=0, inplace=True)
userIDs = user_x_movie.columns
user_profile = pd.DataFrame(columns = movie_profile.columns)
user_x_movie


# Making a pivot table as it will be beneficial in the future as the rows contain ratings for the movie.<br>
# So, if two people like a movie and have rated it similarly, we can just traverse the rows to find similar users.<br>
# Then we can traverse the column to recommend movies to the users

# In[7]:


df_merge_rating_movie = pd.merge(df_movies, df_ratings, on='movieId')
df_merge_rating_movie = df_merge_rating_movie.drop('timestamp',axis=1)
df_merge_rating_movie.head()


# In[8]:


df_merge_tags_movies = pd.merge(df_movies, df_tags, on='movieId')
df_merge_tags_movies = df_merge_tags_movies.drop('timestamp',axis=1)
df_merge_tags_movies.head()


# In[9]:


ratings = pd.DataFrame(df_merge_rating_movie.groupby('title')['rating'].mean())
ratings['count'] = pd.DataFrame(df_merge_rating_movie.groupby('title')['rating'].count())
ratings.head()
ratings2 = ratings[ratings['count'] >= 35].sort_values(by = 'count', ascending = False)
ratings2.head()


# Just taking the mean of movies makes no sense since if only one person has seen it, and rates it highly, it will skew the data. <br>
# So we incorporate that it must be reviewed by n number of viewers so that it's mean rating is fine to be considered.<br>
# n here is 35 which is the average number of reviewsa movie has

# In[10]:


# plt.figure(figsize=(10,4))
# ratings['count'].hist(bins=15)
# plt.ylabel('no. of movies')
# plt.xlabel('counts')
# plt.show()


# most movies have less than 50 ratings and as stated above, the average value is 35. 

# In[11]:


# plt.figure(figsize=(10,4))
# ratings['rating'].hist(bins=50)
# plt.ylabel('no. of movies')
# plt.xlabel('mean ratings')
# plt.show()


# This shows the distribution of movie ratings and how many movies have a ceratin rating

# In[12]:


# sns.jointplot(x='rating',y='count',data=ratings,alpha=0.5)
# plt.show()


# Although, this doesnt convey much,<br> the values on the top right are the best movies because they have a lot of ratings and at the same time it is rated highly by users.<br>
# Bottom right are movies that have a high rating but very few ratings so it cant be considered as a good movie<br>
# Most of the values are in the middle and upper middle. Upper middle are good movies but there is a significant divide between thsoe who like it and those who dont, so must be careful when recommending these movies.

# In[13]:


# plt.figure(figsize=(20,7))
# generlist = df_merge_rating_movie['genres'].apply(lambda generlist_movie : str(generlist_movie).split("|"))
# geners_count = {}

# for generlist_movie in generlist:
#     for i in generlist_movie:
#         if(geners_count.get(i, False)):
#             geners_count[i]=geners_count[i]+1
#         else:
#             geners_count[i] = 1       
# geners_count.pop("(no genres listed)")
# plt.bar(geners_count.keys(),geners_count.values(),color = 'b')
# plt.ylabel('no. of movies')
# plt.xlabel('genres')
# plt.show()


# So, the next step will be to choose the right model to perform user and item based recommendationns

# In[14]:


# plt.figure(figsize=(20,7))
# ratings_grouped_by_movies = df_merge_rating_movie.groupby('title').agg([np.mean, np.size])
# ratings_grouped_by_movies.shape
# ratings_grouped_by_movies['rating']['size'].sort_values(ascending=False).head(10).plot.bar( figsize = (10,5))
# plt.title('Movies with most number of ratings')
# plt.show()


# In[15]:


# ratings_grouped_by_users = df_merge_rating_movie.groupby('userId').agg([np.size, np.mean])
# ratings_grouped_by_users = ratings_grouped_by_users.drop('movieId', axis = 1)
# ratings_grouped_by_users['rating']['size'].sort_values(ascending=False).head(10).plot.bar(figsize = (10,5))
# plt.title('Users who gave most number of ratings')
# plt.show()


# ## Item based recomendation system.

# In[16]:


from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity


# In[17]:


def encode(series, encoder):
    return encoder.fit_transform(series.values.reshape((-1, 1))).astype(int).reshape(-1)

user_encoder, movie_encoder = OrdinalEncoder(), OrdinalEncoder()
df_merge_rating_movie['user_id_encoding'] = encode(df_merge_rating_movie['userId'], user_encoder)
df_merge_rating_movie['movie_id_encoding'] = encode(df_merge_rating_movie['movieId'], movie_encoder)

matrix = csr_matrix((df_merge_rating_movie['rating'], (df_merge_rating_movie['user_id_encoding'], df_merge_rating_movie['movie_id_encoding'])))


# In[18]:


df_merge_rating_movie.head()


# In[19]:


df_matrix = pd.DataFrame(matrix.toarray())
df_matrix


# In[20]:


#Normalizing the matrix
#Rows represent Users
#Columns represent Movies


# In[21]:


df_matrix = df_matrix.sub(df_matrix.sum(axis=1)/df_matrix.shape[1],axis=0)


# In[22]:


cosine_matrix = cosine_similarity(df_matrix.T, df_matrix.T)
movie_encoder.inverse_transform([[1]])


# In[23]:


title_list = df_merge_rating_movie.groupby('title')['movieId'].agg('mean')


# In[24]:


offline_results = {
    movie_id: np.argsort(similarities)[::-1]
    for movie_id, similarities in enumerate(cosine_matrix)
}
class recc:
    def get_recommendations(self,movie_title, top_n):
        movie_id = title_list[movie_title]
        movie_csr_id = movie_encoder.transform([[movie_id]])[0, 0].astype(int)
        rankings = offline_results[movie_csr_id][:top_n]
        ranked_indices = movie_encoder.inverse_transform(rankings.reshape((-1, 1))).reshape(-1)
        return df_movies.set_index('movieId').loc[ranked_indices]


# In[25]:


a = recc()
a.get_recommendations('Matrix, The (1999)', 10)


# In[26]:


import pickle
pickle_out = open('recc.pkl', 'wb')
pickle.dump(a, pickle_out)
pickle_out.close()


# In[ ]:




