import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000/query"

# Page config
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("💼 HR Policy Chatbot")
st.write("Ask any question about your company's HR policies:")

query = st.text_input("Your question:")

if st.button("Get Answer") and query:
    st.markdown(f"**You:** {query}")

    payload = {"query": query}
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer found.")
            st.markdown(f"**HR Bot:** {answer}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to the backend: {str(e)}")
