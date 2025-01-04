import streamlit as st
import gensim
from gensim import corpora, models
import re
import shutil
import pandas as pd


def perform_topic_modeling(transcript_text, num_topics):
    # Preprocess the text
    text = transcript_text
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Prepare the dataset
    sentences = re.split(r'\.|\!|\?', text)
    sentences = [re.sub(r'\W', ' ', sent) for sent in sentences]
    sentences = [re.sub(r'\s+', ' ', sent) for sent in sentences]
    sentences = [sent for sent in sentences if sent != '']

    # Tokenize the dataset
    words = [sent.split() for sent in sentences]

    # Create the dictionary
    dictionary = corpora.Dictionary(words)

    # Create the corpus
    corpus = [dictionary.doc2bow(word) for word in words]

    # Perform the LDA
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    return lda_model

st.set_page_config(layout='wide')
Choice = st.sidebar.selectbox("Select your choice", ["on Text", "On Video", "On CSV"])

if Choice ==  "on Text":

    st.subheader("Topic Modeling and Labeling App")

    text_input =st.text_area("Paste your text here", height=400)

    if text_input is not None:
        if st.button("Analyze Text"):
           col1, col2,col3 = st.columns([1,1,1])
           with col1:
               st.info("Text is below")
               st.success(text_input)
            
      

