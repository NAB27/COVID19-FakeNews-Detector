import archModel as am
import streamlit as st
import pickle
import torch


from .cleanData import hard_cleaning, remove_en_stop_words, text_lemmatizer

with open("ressources/tfidf.pkl", 'rb') as f:
    tfidf = pickle.load(f)

model = am.NN2()
model.load_state_dict(torch.load('ressources/model_weights.pth'))
model.eval()

# fonction qui fait le nettoyage, suppression des SW ,et lemmatisation pour un texte donné en paramétre 
def text_to_corpus(text):
  text = hard_cleaning(text)
  text = remove_en_stop_words(text)
  text = text_lemmatizer(text)
  return text 


def app():
    
    with st.container():
        st.subheader("Test our model :")
        input = st.text_input("Add a tweet")
        bt_clean = st.button("Apply")

        if bt_clean:
            ttc = text_to_corpus(input)
            data_to_pred = torch.tensor(tfidf.transform([ttc]).toarray())
            prediction = model(data_to_pred.float())
            result = torch.argmax(prediction)
            if result == torch.tensor(0):
                st.write('This tweet is : Fake')
            else :
                st.write('This tweet is : Real')
            
