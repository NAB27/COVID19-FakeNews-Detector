from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import streamlit as st
from PIL import Image
import pickle
import nltk
import re 


with open('ressources\data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('ressources\corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

with open('ressources\dtm.pkl', 'rb') as f:
    dtm = pickle.load(f)

with open('ressources\labels.pkl', 'rb') as f:
    labels = pickle.load(f)


# cleaning and preprocessing functions 
# step 1 
def hard_cleaning(text):
  text = text.lower() 
  text = re.sub(r'http\S+', '', text) 
  text = re.sub('[^a-zA-Z]+', ' ', text)
  return text

# step 2 
stop_words = set(stopwords.words('english'))
def remove_en_stop_words(text):
    words = text.split()
    noise_free_words = [word for word in words if word not in stop_words] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

# step 3
lemmatizer = WordNetLemmatizer()
# POS_TAGGER_FUNCTION : TYPE 1
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None
# fonction de limmatization de text 
def text_lemmatizer(text):
  pos_tagged = nltk.pos_tag(nltk.word_tokenize(text)) 
  wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
  lemmatized_sentence = []
  for word, tag in wordnet_tagged:
      if tag is None:
          # if there is no available tag, append the token as is
          lemmatized_sentence.append(word)
      else:       
          # else use the tag to lemmatize the token
          lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
  lemmatized_sentence = " ".join(lemmatized_sentence)
  return lemmatized_sentence






def app():

    with st.container():
        st.subheader("Samples from original data :")
        st.table(data.head())


    with st.container():
        st.subheader("Samples from cleaned data :")
        st.table(corpus.head())

    with st.container():
        st.subheader("Steps of Data cleaning process :")
        option = st.selectbox(
        'Any step would you like to discover ?',
        ('Keep only letters', 'Remove stop words', 'Lemmatization'))

        input = st.text_input("Add text to test with")

        bt_clean = st.button("Apply")

        if option == 'Keep only letters' :
            if bt_clean:
                result = hard_cleaning(input)
                st.write('Result : ', result)
            
                
        elif option == 'Remove stop words' :
            if bt_clean:
                result = remove_en_stop_words(input)
                st.write('Result : ', result)

        else :
            if bt_clean:
                result = text_lemmatizer(input)
                st.write('Result : ', result)


    with st.container():
        st.subheader("Words cloud for real & fake tweets :")

        real_wc = Image.open("ressources/real_wc.png")
        st.image(real_wc, use_column_width=True)
        st.caption("words cloud for real tweets")

        fake_wc = Image.open("ressources/fake_wc.png")
        st.image(fake_wc, use_column_width=True)
        st.caption("words cloud for fake tweets")
    
    with st.container():
        st.subheader("Preparing Data for models :")
        st.markdown("From corpus to Document-Term Matrix (After applying TF-IDF on cleaned data)") 
        st.dataframe(dtm.head())
        st.markdown("Labels normalization")
        st.dataframe(labels[:5])