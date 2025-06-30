import streamlit as st
from helper import get_conversational_chain, SupaBaseChatHistory


st.title("🩺 MediBot AI — Ask Health Questions, Get Trusted Answers")

conversational_chain = get_conversational_chain()

if "session_id" not in st.session_state:
    st.session_state.session_id = "single-session"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a medical question...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = SupaBaseChatHistory(st.session_state.session_id)
    history.add_user_message(prompt)

    with st.spinner("Thinking..."):
        # Run chain → it pulls history from Supabase
        response = conversational_chain.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        output = response["output"]

    st.session_state.messages.append({"role": "assistant", "content": output})
    with st.chat_message("assistant"):
        st.markdown(output)

    history.add_ai_message(output)
