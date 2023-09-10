import streamlit as st
from streamlit_chat import message

# from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import base64

st.set_page_config(layout="wide")
load_dotenv()
query = None

def init():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 50%;
            max-width: 50%;
        }
        """,
        unsafe_allow_html=True,
    )


def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    st.session_state.generated.append("The messages from Bot\nWith new line")


def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]


st.session_state.setdefault("generated", [])


def save_uploadedfile(uploadedfile):
    with open(os.path.join("", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())


def displayPDF(file, placeholder=st):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}"  type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def app():
    query = None
    with st.sidebar:
        pdf = st.file_uploader("", type=["pdf"])
        placeholder = st.empty()
    with st.container():
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        if pdf is not None:
            pdf_text = PdfReader(pdf)

            text = ""
            for page in pdf_text.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100, length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            file_name = pdf.name[:-4]

            if os.path.exists(file_name + ".pk1"):
                with open(file_name + ".pk1", "rb") as f:
                    vectorstore = pickle.load(f)
            else:
                with open(file_name + ".pk1", "wb") as f:
                    emdeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_texts(chunks, emdeddings)
                    pickle.dump(vectorstore, f)

            save_uploadedfile(pdf)

            with open(pdf.name, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")

                # Embedding PDF in HTML
                pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="900px" type="application/pdf">'

                # Displaying File
                col1.markdown(pdf_display, unsafe_allow_html=True)

            query = col4.text_input("Enter your question here")

            if query:
                docs = vectorstore.similarity_search(query, topn=5)
                llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    # print(cb)
                st.session_state["generated"].append({"role": "user", "content": query})
                st.session_state["generated"].append(
                    {"role": "assistant", "content": response}
                )
                for element in st.session_state["generated"]:
                    role = True if element["role"] == "user" else False
                    with col2.container():
                        message(element["content"], is_user=role)

app()