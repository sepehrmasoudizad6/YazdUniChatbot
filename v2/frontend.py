import streamlit as st
import requests

st.title("دستیار هوشمند دانشگاه یزد")

# Initialize session state for query and response
if "query" not in st.session_state:
    st.session_state.query = ""
if "response" not in st.session_state:
    st.session_state.response = None

# Input field for user query
query = st.text_input("سوال خود را وارد کنید:", value=st.session_state.query)

# Button to submit query
if st.button("ارسال"):
    if query:
        try:
            # Send request to FastAPI backend
            response = requests.post(
                "http://localhost:8000/query",
                json={"query": query}
            )
            response.raise_for_status()
            st.session_state.response = response.json()
            st.session_state.query = query
        except requests.exceptions.RequestException as e:
            st.error(f"خطا در ارتباط با سرور: {str(e)}")
    else:
        st.warning("لطفاً یک سوال وارد کنید.")

# Display response if available
if st.session_state.response:
    st.subheader("پاسخ:")
    st.write(st.session_state.response["answer"])
    
    st.subheader("منابع:")
    for i, context in enumerate(st.session_state.response["context"], 1):
        with st.expander(f"منبع {i}"):
            st.write(context)

# Clear button
if st.button("پاک کردن"):
    st.session_state.query = ""
    st.session_state.response = None
    st.rerun()