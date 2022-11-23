import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from Algor import Algor

movies = pkl.load(open('./data/movies.pkl','rb'))
genome = pkl.load(open('./data/genome.pkl','rb'))
# tags = pkl.load(open('./data/tags.pkl','rb'))
weights = pkl.load(open('./data/weights.pkl', 'rb'))
# with open("./data/list_movies.pkl", "rb") as f:
#     ListMovies = pkl.load(f)
posters = pd.read_csv("./posters.csv")
direct = pd.read_csv("./direc_actor.csv")
plot = pd.read_csv("./plot.csv")
tag_genome = Algor(movies, genome, weights, posters, direct, plot)


st.title('Movie Recommendation System')
movie = st.selectbox("Movie selection: ", options = list(tag_genome.ListMovies))
# movie = st.text_input('Enter selection: ')
# AnalystButton = st.button('Analyst')

with st.container():
    des1, des2 = st.columns(2)
    with des1:
        st.image('https://images-na.ssl-images-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_UX182_CR0,0,182,268_AL_.jpg')
    with des2:
        st.write('alol')    
col1, col2, col3, col4, col5 = st.columns(5)
ListDirec = []
ListTag = ['Action', 'Violence', 'Funny', 'Fantasy', 'Comedy', 'Classic', 'Romance', 'Sci-fi',  'Atmospheric', 'Based On A Book']
Lv = {
    'Less' : -0.5,
    'Ok'   : 1,
    'More' : 0.5
}
Choice = ['Less', 'Ok', 'More']
# with col1: 
#     for tag in ListTag:
#         st.text(tag)
# with col2:
#     for tag in ListTag:
#         globals()[tag + ' Bar'] = st.progress(0)
#         if movie :
#             globals()[tag + ' Bar'].progress(rel_item(movie, tag.lower()))
with col1: 
    st.subheader('Action')
    ActionBar = st.progress(0)
    if movie :
        ActionBar.progress(tag_genome.rel_item(movie, ListTag[0].lower()))

    ActionLv = st.radio('', Choice, key= 'Action', index= 1, label_visibility='hidden')
    ListDirec.append(('action', Lv.get(ActionLv)))

    st.subheader('Violence')
    ViolenceBar = st.progress(0)
    if movie :
        ViolenceBar.progress(tag_genome.rel_item(movie, ListTag[1].lower()))

    ViolentLv = st.radio('', Choice, key= 'Violence', index= 1, label_visibility='hidden')
    ListDirec.append(('violence', Lv.get(ViolentLv)))

with col2:
    st.subheader('Funny')
    FunnyBar = st.progress(0)
    if movie :
        FunnyBar.progress(tag_genome.rel_item(movie, ListTag[2].lower()))

    FunnyLv = st.radio('', Choice ,key= 'Funny', index= 1, label_visibility='hidden')
    ListDirec.append(('funny', Lv.get(FunnyLv)))

    st.subheader('Fantasy')
    FantasyBar = st.progress(0)
    if movie :
        FantasyBar.progress(tag_genome.rel_item(movie, ListTag[3].lower()))

    FantasyLv = st.radio('', Choice,key= 'Fantasy', index= 1, label_visibility='hidden')
    ListDirec.append(('fantasy', Lv.get(FantasyLv)))

with col3: 
    st.subheader('Comedy')
    ComedyBar = st.progress(0)
    if movie :
        ComedyBar.progress(tag_genome.rel_item(movie, ListTag[4].lower()))

    ComedyLv = st.radio('', Choice ,key= 'Comedy', index= 1, label_visibility='hidden')
    ListDirec.append(('comedy', Lv.get(ComedyLv)))

    st.subheader('Classic')
    ClassicBar = st.progress(0)
    if movie :
        ClassicBar.progress(tag_genome.rel_item(movie, ListTag[5].lower()))

    ClassicLv = st.radio('', Choice,key= 'Classic', index= 1, label_visibility='hidden')
    ListDirec.append(('classic', Lv.get(ClassicLv)))
with col4: 
    st.subheader('Romance')
    RomanceBar = st.progress(0)
    if movie :
        RomanceBar.progress(tag_genome.rel_item(movie, ListTag[6].lower()))

    RomanceLv = st.radio('', Choice,key= 'Romance', index= 1, label_visibility='hidden')
    ListDirec.append(('romance', Lv.get(RomanceLv)))

    st.subheader('Sci-fi')
    ScifiBar = st.progress(0)
    if movie :
        ScifiBar.progress(tag_genome.rel_item(movie, ListTag[7].lower()))

    ScifiLv = st.radio('', Choice,key= 'Sci-fi', index= 1, label_visibility='hidden')
    ListDirec.append(('sci-fi', Lv.get(ScifiLv)))
with col5:
    st.subheader('Atmospheric')
    AtmosphericBar = st.progress(0)
    if movie :
        AtmosphericBar.progress(tag_genome.rel_item(movie, ListTag[8].lower()))

    AtmosphericLv = st.radio('', Choice ,key= 'Atmospheric', index= 1, label_visibility='hidden')
    ListDirec.append(('atmospheric', Lv.get(AtmosphericLv)))

    st.subheader('BaseOnBook')
    BaseOnBookBar = st.progress(0)
    if movie :
        BaseOnBookBar.progress(tag_genome.rel_item(movie, ListTag[9].lower()))

    BaseOnBookLv = st.radio('', Choice ,key= 'BaseOnBook', index= 1, label_visibility='hidden')
    ListDirec.append(('based on a book', Lv.get(BaseOnBookLv)))

if st.button('Recommend movie'):
    result = tag_genome.recommendation(movie, ListDirec)
    for movie in result:
        st.write(movie[0])
        try:
            with st.container():
                cols1, cols2 = st.columns(2)
                with cols1:
                    st.image(tag_genome.extract_urls(movie[0]))
                with cols2:
                    st.write('Directors :', tag_genome.extract_direct(movie[0])[0])
                    st.write('Actors :', tag_genome.extract_direct(movie[0])[1])
                    st.write('Plot :', tag_genome.extract_plot(movie[0]))
        except:
            pass