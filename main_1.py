import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

# 🔑 Load LLM (Chat Model)
def load_chat_model(openai_api_key):
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini"  # Or "gpt-3.5-turbo"
    )
    return llm

# 🎨 UI Setup
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

col1, col2 = st.columns(2)
with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")
with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")

# 🔐 Input OpenAI API Key
st.markdown("## Enter Your OpenAI API Key")
def get_openai_api_key():
    return st.text_input(
        label="OpenAI API Key",
        placeholder="Ex: sk-2twmA8tfCb8un4...",
        key="openai_api_key_input",
        type="password"
    )

openai_api_key = get_openai_api_key()

# 📂 File Input
st.markdown("## Upload the text file you want to summarize")
uploaded_file = st.file_uploader("Choose a file", type="txt")

# 📄 Summary Output
st.markdown("### Here is your Summary:")

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_input = stringio.read()

    if len(file_input.split(" ")) > 20000:
        st.write("Please enter a shorter file. The maximum length is 20000 words.")
        st.stop()

    if file_input:
        if not openai_api_key:
            st.warning(
                'Please insert OpenAI API Key. '
                'Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)',
                icon="⚠️"
            )
            st.stop()

    # ✂️ Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=5000,
        chunk_overlap=350
    )
    splitted_documents = text_splitter.create_documents([file_input])

    # 🤖 Load Chat Model and Chain
    llm = load_chat_model(openai_api_key=openai_api_key)
    summarize_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")

    # 🧠 Generate Summary
    summary_output = summarize_chain.invoke(splitted_documents)
    st.markdown("### 📄 Summary:")
    st.success(summary_output["output_text"])

    #st.write(summary_output)
