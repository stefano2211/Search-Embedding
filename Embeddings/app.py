import streamlit as st
import pandas as pd
import numpy as np
import openai
import config
from openai.embeddings_utils import get_embedding, cosine_similarity

openai.api_key = config.OPENAI_API_KEY
    

#Home page
microsoft_news = pd.read_csv('dataset/news.csv')
st.title('Search Embeddings with Microsoft Earnings Call Transcript')
st.write('The application s search system is based on gpt chat to detect similarities between what the user enters and the data already loaded into the application.')
st.write('In the application you can search for a sentence related to microsoft and it will make a relation with the dataset that is already loaded and vectorized to find the news most related to what the user entered.')

with st.form(key='search'):
    search = st.text_input('Find Search')
    submit_search = st.form_submit_button(label='Search')
if submit_search:
    search_term_vector = get_embedding(search, engine="text-embedding-ada-002")
    microsoft_news['embeddings'] = microsoft_news['embeddings'].apply(eval).apply(np.array)
    microsoft_news["similarities"] = microsoft_news['embeddings'].apply(lambda x: cosine_similarity(x, search_term_vector))
    sorted_by_similarity = microsoft_news.sort_values("similarities", ascending=False).head(3)
    results = sorted_by_similarity['text'].values.tolist()
    for results in results:
        st.write(results)



