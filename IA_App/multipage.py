import streamlit as st
from PIL import Image


class MultiPage: 

    def __init__(self) -> None:
        self.pages = []
    
    def add_page(self, title, func) -> None: 

        self.pages.append({
                "title": title, 
                "function": func
            })

    def run(self):

        page = st.sidebar.selectbox(
            'Navigation Bar : ', 
            self.pages, 
            format_func=lambda page: page['title']
        )
        
        page['function']()

        for i in range(18):
            st.sidebar.text('')

        st.sidebar.text("Fake Tweets Detection App made by :") 

        col1, col2 = st.sidebar.columns([1,3])
        with col1:
            twitter = Image.open("ressources/twitter.png")
            st.image(twitter)

        with col2:
            st.caption("BEN SALAH Abderrahman & MOCHIR Nabih")