import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# --- Local Embedding Model Load ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Streamlit Setup ---
st.set_page_config(page_title="Simple PDF Q&A", page_icon="📄", layout="wide")
st.title("📄 PDF Question Generator & Chat Agent")

# --- Session State ---
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 ১. সেটিংসে তথ্য দিন")
    groq_api_key = st.text_input("Groq API Key (gsk-...)", type="password")
    
    # Model Selection Dropdown
    selected_model = st.selectbox(
        "Groq Model নির্বাচন করুন",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
    )

    uploaded_pdf = st.file_uploader("PDF ফাইল আপলোড করুন", type=["pdf"])

    if uploaded_pdf and groq_api_key and st.button("PDF প্রসেস করুন"):
        with st.spinner("PDF পড়া হচ্ছে..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            embeddings = load_embeddings()
            vectorstore = FAISS.from_documents(splits, embeddings)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("PDF প্রসেসিং সম্পন্ন হয়েছে!")

# --- Main Interface ---
if st.session_state.retriever:
    llm = ChatGroq(
        model=selected_model,
        groq_api_key=groq_api_key.strip(),
        temperature=0.3
    )

    # Auto Question Generation Button
    if st.button("💡 PDF থেকে ৫টি গুরুত্বপূর্ণ প্রশ্ন বের করুন"):
        with st.spinner("প্রশ্ন তৈরি করা হচ্ছে..."):
            docs = st.session_state.retriever.invoke("main topics and key points")
            context = "\n".join([d.page_content for d in docs])
            
            gen_prompt = f"Based on the following text, extract 5 important questions in Bengali that can be answered using this content:\n\n{context}"
            questions = llm.invoke(gen_prompt).content
            st.info(f"**PDF থেকে সম্ভাব্য প্রশ্নসমূহ:**\n\n{questions}")

    st.divider()

    # Chat UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("PDF সম্পর্কে প্রশ্ন করুন...")

    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.spinner("উত্তর খোঁজা হচ্ছে..."):
            retrieved_docs = st.session_state.retriever.invoke(user_query)
            context_str = "\n\n".join([d.page_content for d in retrieved_docs])

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "আপনি একজন সহায়ক সহকারী। প্রদত্ত কন্টেন্ট ব্যবহার করে প্রশ্নের উত্তর দিন। যদি উত্তর না থাকে তবে বলুন 'তথ্যটি PDF-এ নেই।'\n\nকন্টেন্ট:\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])

            chain = qa_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": context_str,
                "chat_history": st.session_state.chat_history,
                "input": user_query
            })

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=response)
        ])
else:
    st.info("👈 সাইডবারে Groq API Key দিন এবং একটি PDF আপলোড করে প্রসেস বাটনে ক্লিক করুন।")
    
