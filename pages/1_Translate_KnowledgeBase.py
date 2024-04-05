import streamlit as st 
import Libs as glib 

st.set_page_config(page_title="Translate with knowledge base")
input_text = st.text_input("Search Knowledge base") 
if input_text: 
    response = glib.search(input_text) 
    st.write(response["result"])
    st.write(response)
    
