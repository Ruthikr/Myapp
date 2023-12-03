import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.set_page_config(layout="wide")
st.title("Enter the transaction details")
st.write("The model will predict whether they are fraud or not")
pipe=pickle.load(open("bank_fraud_pipeline.pkl","rb"))


with st.form(key='form'):
    transaction_amount=st.number_input(label="enter transaction_amount")
    country=st.text_input(label="enter country")
    corporation=st.text_input(label="enter corporation")
    Age=st.number_input(label="enter age")
    gender=st.selectbox('select gender',('M','F'))
    st.write('You selected:',gender)
    submit_button = st.form_submit_button(label='Result')


user_input=np.array([transaction_amount,country,corporation,Age,gender],dtype=object).reshape(1,5)
result=pipe.predict(user_input)
if st.form !=null:
    if submit_button:
        if result[0]==1:
            label=" **Fraud** "
            st.error(label)
        else:
            st.success("**not fraud**")
