import streamlit as st
from PIL import Image

def app():
    
    with st.container():
        st.subheader("The training Process for an ANN :")

        trainingProcessImage = Image.open("ressources/trainingProcess.png")
        st.image(trainingProcessImage , use_column_width=True)
        st.caption("Source : deep learning with Pytorch")   

    with st.container():
        st.subheader("The steps of training phase :")

        st.markdown("###### Step 1 : Weights initialization")
        st.markdown("###### Step 2 : Forward propagation")
        st.markdown("###### Step 3 : Backward propagation")
        st.markdown("###### Step 4 : Update weights")
        st.caption("PS : The whole process will be repeted N times")

    with st.container():
        st.subheader("Models architectures :")

        modelsArch = Image.open("ressources/modelsArch.png")
        st.image(modelsArch , use_column_width=True)
        st.caption("Figure represent models architectures")   

    with st.container():
        st.subheader("Our choice :")

    col1, col2 = st.columns([2,2])

    with col1:
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("We have chosen the model with the highest precision the 6th model from our catalog")

    with col2:
        st.caption("red line for train loss - green line for val loss")
        loss = Image.open("ressources/losses.png")
        st.image(loss)
        


        