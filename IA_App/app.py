from enum import auto
import streamlit as st
from PIL import Image

from multipage import MultiPage
from pages import cleanData, testModel, trainingProcess

app = MultiPage()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ensias = Image.open("ressources/ensias.png")
    st.image(ensias)

with col5:
    um5 = Image.open("ressources/um5.png")
    st.image(um5)


st.title("Welcome to our lovely application - Fake tweet detection !")

app.add_page("Data Cleaning & Preprocessing ", cleanData.app)
app.add_page("Training process", trainingProcess.app)
app.add_page("Model testing", testModel.app)



app.run()