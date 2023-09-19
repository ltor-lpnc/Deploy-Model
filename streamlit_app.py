import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('rasmussen_model.sav','rb'))
st.title("Prediction of status")
st.markdown("We use Synt and Phon scores to predict status")

st.subheader("Enter Synt Score : ")
synt = st.number_input('', 0,1,key='synt')

st.subheader("Enter Phon Score : ")
phon = st.number_input('', 0,1,key='phon')

st.subheader("Status prediction :")
X = np.array([synt,phon]).reshape(1, -1)
st.code(float(model.predict(X)))
