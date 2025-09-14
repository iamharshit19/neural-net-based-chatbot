import streamlit as st
import random
import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()


# --- Function to load model and data ---
# Using st.cache_resource to load these only once
@st.cache_resource
def load_chatbot_essentials():
    """Loads the pre-trained model, words, and classes."""
    
    try:
      nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
    return intents, words, classes, model

# Check if model and data files exist. If not, show an error.
if not all([os.path.exists(f) for f in ['intents.json', 'words.pkl', 'classes.pkl', 'chatbot_model.h5']]):
    st.error("Model and data files not found. Please run the `training.py` script first to generate them.")
else:
    # Load the files
    intents, words, classes, model = load_chatbot_essentials()

    # --- Chatbot Core Functions ---
    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(intents_list, intents_json):
        if not intents_list:
            return "I'm sorry, I don't understand. Could you please rephrase?"
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tags'] == tag:
                result = random.choice(i['responses'])
                break
        else:
            result = "I'm not sure how to respond to that."
        return result


    # --- Streamlit UI ---

    st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")

    st.title("🤖 Simple Chatbot")
    st.write("This is a simple chatbot powered by a neural network. Ask me anything!")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! How can I help you today?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ints = predict_class(prompt)
                response = get_response(ints, intents)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
