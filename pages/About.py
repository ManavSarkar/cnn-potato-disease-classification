import streamlit as st

def About():
    st.title("About")
    st.write("This web application is developed by Manav Sarkar.")
    st.write("Github Profile: [Manav Sarkar](https://github.com/ManavSarkar)")
    st.write("Project Link: [Potato Leaf Disease Classification](https://github.com/ManavSarkar/cnn-potato-disease-classification)")
    st.write("Technologies Used: Convolutional Neural Network (CNN)")
    st.write("")
    st.write("### Project Description:")
    st.write("Potato Leaf Disease Classification is a deep learning project aimed at classifying different types of diseases that affect potato plants based on images of their leaves.")
    st.write("The model is built using a Convolutional Neural Network (CNN), a type of deep learning model well-suited for image classification tasks. It has been trained on a dataset containing images of healthy potato leaves as well as leaves affected by various diseases.")
    st.write("Once trained, the model can accurately predict the type of disease affecting a potato plant based on an image of its leaf, providing valuable insights to farmers and researchers for early detection and management of diseases.")
    st.write("")

About()
