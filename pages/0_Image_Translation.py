import streamlit as st
import Libs as glib

st.title("Image Translation")


uploaded_file = st.file_uploader("Select an image", type=['png', 'jpg'], label_visibility="collapsed")
go_button = st.button("Go", type="primary")
if go_button and uploaded_file:
    st.image(glib.get_bytesio_from_bytes(uploaded_file.getvalue()))
    image_bytes = uploaded_file.getvalue()
            
    response = glib.get_response_from_model(
        image_bytes=image_bytes,
    )
    st.write(response)
