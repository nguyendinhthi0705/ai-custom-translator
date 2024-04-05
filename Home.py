import streamlit as st 
import Libs as glib 
import json

st.set_page_config(page_title="Home")


input_text = st.text_input("Input your text") 
if input_text: 
    with st.chat_message("user"): 
        st.markdown(input_text) 
    response = glib.call_claude_sonet_stream(input_text)
    st.write_stream(response)

    



    
   