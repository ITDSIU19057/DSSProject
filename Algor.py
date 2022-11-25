import pandas as pd 
import numpy as np
from tqdm import tqdm
import json
import streamlit as st

@st.cache
class Algor():
    def __init__(self, movies, genome, weights, poster, direct, plot):
        self.np_genome = genome.to_numpy()
        self.np_movies = movies.to_numpy()
        self.np_posters = poster.to_numpy()
        self.np_directs = direct.to_numpy()
        self.np_plot = plot.to_numpy()
        self.weights = weights
        self.ListMovies = self.np_movies[:,1]
        self.len_movie = len(self.np_movies)
        self.movies_id, self.tags_id = self.create_id()
        self.movies_encode = self.encode_movies()

    def create_id(self):
        movies_id = {}
        for index, movie in enumerate(self.np_movies):
            movies_id[movie[1]] = (movie[0], index)
        tags_id = {}
        for index, tag in enumerate(np.unique(self.np_genome[:,3])):
            tags_id[tag] = (np.unique(self.np_genome[:,1])[index],index)
        return movies_id, tags_id

    def rel_item(self,item,tag):
        movie_id = self.movies_id[item][0]
        tag_id = self.tags_id[tag][0]
        return float(self.np_genome[(np.where((self.np_genome[:,0] == movie_id) & (self.np_genome[:,1] == tag_id)))][:,2])

    def rel_all_items(self, tag):
        tag_index = self.tags_id[tag][1]
        rel_items = self.np_genome[tag_index*self.len_movie:tag_index*self.len_movie+self.len_movie,2].tolist()
        return rel_items

    def linear_sat(self,item,tag,direction):
        a = (self.rel_all_items(tag) - np.array(self.rel_item(item,tag)))*direction
        result = np.clip(a, 0, 1)
        return result

    def diminish_sat(self,item,tag,direction):
        critique_dist = self.linear_sat(item,tag,direction)
        result = 1 - np.exp(-5*critique_dist)
        return result

    def tag_item_vector(self, item):
        item_id = self.movies_id[item][0]
        full_tags_item = [self.np_genome[i][2] for i in np.where(self.np_genome[:,0] == item_id)[0]]  
        return full_tags_item

    def tag_item_vector(self, item):
        item_id = self.movies_id[item][0]
        full_tags_item = [self.np_genome[i][2] for i in np.where(self.np_genome[:,0] == item_id)[0]]  
        return full_tags_item

    def cos_similarity(self, item, weights):
        
        # Create all movies's tags scores
        list_tags = np.zeros((self.len_movie,len(self.tags_id)))

        for index,tag_score in enumerate(self.np_genome[:,2]):
            tags_index = index // self.len_movie
            movies_index = index - tags_index*self.len_movie
            list_tags[movies_index,tags_index] = tag_score
            
        # Extract 1 query movie from matrix above 
        query_index = self.movies_id[item][1]
        query_tags = list_tags[query_index]
        
        numerator = np.dot(list_tags * list_tags[query_index],weights)
        
        denominator = np.sqrt(np.dot(list_tags**2, weights)) * np.sqrt(np.dot(query_tags**2, weights))
        
        result = numerator / denominator
        return result

    def encode_movies(self):
        genres = {}
        index = 0
        movies_encode = []
        for movie in self.np_movies[:,2]:
            movie_encode = [0]*20
            for genre in movie.split('|'):
                if genre not in genres:
                    genres[genre] = index
                    index += 1
                movie_encode[genres[genre]] = 1
            movies_encode.append(movie_encode)
        return movies_encode

    def genres_similarity(self, item):
        index = self.movies_id[item][1]
        numerator = np.dot(self.movies_encode, self.movies_encode[index])
        denominator = np.sqrt(np.square(self.movies_encode[index]).sum()) * np.sqrt(np.sum(np.square(self.movies_encode),axis=1))
        return numerator/denominator

    def recommendation(self, item, list_critiques):
        score = 0

        if item not in self.movies_id:
            print('Have not the film in system')
            return

        for critique in list_critiques:
            tag = critique[0]
            direction = critique[1]
            value = self.diminish_sat(item,tag,direction)
            cos = self.cos_similarity(item, self.weights)
            genre_simi = self.genres_similarity(item)
            score += value * cos * genre_simi
        score_dict = {}
        for index, movie in enumerate(self.np_movies[:,1]):
            score_dict[movie] = score[index]   
        result = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)[:10]
        return result

    def extract_urls(self, movie):
        movie_id = self.movies_id[movie][0]

        url = self.np_posters[np.where(self.np_posters[:,0] == movie_id)[0]][0,2]
        return url

    def extract_direct(self, movie):
        movie_id = self.movies_id[movie][0]
        
        direct = self.np_directs[np.where(self.np_directs[:,0] == movie_id)[0]][0,1]
        actor = self.np_directs[np.where(self.np_directs[:,0] == movie_id)[0]][0,2]
        return (direct, actor)

    def extract_plot(self, movie):
        movie_id = self.movies_id[movie][0]
    
        plot = self.np_plot[np.where(self.np_plot[:,0] == movie_id)[0]][0,1]
        return plot
    
    def __call__(self, item, list_critiques):
        return self.recommendation(item, list_critiques)
