import tempfile
import io
import base64
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from fpdf import FPDF
import openai
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# --- Caching local CPU embedding model ---
@st.cache_resource
def load_local_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-2")

# --- Custom Hybrid Retriever ---
class CustomHybridRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        combined_docs = []
        for retriever in self.retrievers:
            combined_docs.extend(retriever.invoke(query))
        
        seen = set()
        unique_docs = []
        for doc in combined_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs[:4]

# --- Streamlit UI Config ---
st.set_page_config(page_title="Free OpenAI Multimodal RAG", page_icon="✨", layout="wide")
st.title("✨ Free Multimodal Hybrid RAG Agent (Powered by OpenAI API)")

# --- Session State Initialization ---
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Helper 1: Extract Diagrams ---
def extract_diagrams_from_page(pdf_path: str, page_number: int):
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        image_list = page.get_images(full=True)
        images = []
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(Image.open(io.BytesIO(image_bytes)))
        return images
    except Exception:
        return []

# --- Helper 2: Generate PDF ---
def generate_simple_pdf(text_content: str) -> io.BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(0, 10, "AI RAG Response Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, txt=text_content)
    
    buffer = io.BytesIO()
    buffer.write(pdf.output())
    buffer.seek(0)
    return buffer

# --- Helper 3: Vision Question Extraction (OpenAI gpt-4o-mini) ---
def extract_question_from_image(pil_image, api_key: str) -> str:
    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        client = openai.OpenAI(api_key=api_key.strip())
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the exact question, problem, or main subject from this image clearly into plain text."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_str}"}
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ছবি পড়তে সমস্যা হয়েছে: {str(e)}"

# --- Sidebar Setup ---
with st.sidebar:
    st.header("🔑 ১. OpenAI API Key ও PDF আপলোড")
    openai_api_key = st.text_input("OpenAI API Key দিন (sk-...)", type="password")

    uploaded_pdf = st.file_uploader("PDF ফাইল আপলোড করুন", type=["pdf"])
    
    if uploaded_pdf and openai_api_key and st.button("PDF প্রসেস করুন"):
        with st.spinner("PDF প্রসেস করা হচ্ছে (Local Embedding)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                st.session_state.pdf_path = tmp_file.name

            loader = PyPDFLoader(st.session_state.pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            splits = text_splitter.split_documents(docs)

            cleaned_splits = []
            if splits:
                for doc in splits:
                    text = doc.page_content.strip() if doc.page_content else ""
                    if text:
                        clean_text = text.encode("utf-8", "ignore").decode("utf-8")
                        doc.page_content = clean_text
                        cleaned_splits.append(doc)

            if not cleaned_splits:
                st.error("PDF থেকে কোনো পড়ার মতো টেক্সট পাওয়া যায়নি।")
            else:
                clean_api_key = openai_api_key.strip()
                
                embeddings = load_local_embeddings()
                
                vectorstore = FAISS.from_documents(cleaned_splits, embeddings)
                vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                bm25_retriever = BM25Retriever.from_documents(cleaned_splits)
                bm25_retriever.k = 3

                hybrid_retriever = CustomHybridRetriever(retrievers=[bm25_retriever, vector_retriever])

                llm = ChatOpenAI(
                    model="gpt-4o-mini", 
                    openai_api_key=clean_api_key,
                    temperature=0
                )

                # Pure LCEL Pipeline (Zero legacy chain dependencies)
                rephrase_system_prompt = (
                    "Given a chat history and the latest user question "
                    "which might reference context in the chat history, "
                    "formulate a standalone question. Do NOT answer the question, "
                    "just rephrase it if needed or return it as is."
                )
                rephrase_prompt = ChatPromptTemplate.from_messages([
                    ("system", rephrase_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])
                rephrase_chain = rephrase_prompt | llm | StrOutputParser()

                qa_system_prompt = (
                    "Answer the question using ONLY the provided context below. "
                    "If context doesn't contain the answer, say 'তথ্যটি PDF-এ পাওয়া যায়নি।'\n\n"
                    "{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])
                qa_chain = qa_prompt | llm | StrOutputParser()

                def run_rag_pipeline(query_text: str, history: List):
                    if history:
                        standalone_q = rephrase_chain.invoke({"input": query_text, "chat_history": history})
                    else:
                        standalone_q = query_text
                    
                    retrieved_docs = hybrid_retriever.invoke(standalone_q)
                    context_str = "\n\n".join(d.page_content for d in retrieved_docs)
                    
                    answer_text = qa_chain.invoke({
                        "context": context_str,
                        "chat_history": history,
                        "input": query_text
                    })
                    return {"answer": answer_text, "context": retrieved_docs}

                st.session_state.rag_pipeline = run_rag_pipeline
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success("Indexing সফল হয়েছে!")

# --- Main Interface ---
st.subheader("২. আপনার প্রশ্ন প্রদান করুন")

uploaded_img = st.file_uploader("প্রশ্ন সম্বলিত ছবি আপলোড করুন (ঐচ্ছিক)", type=["jpg", "jpeg", "png"])
extracted_image_question = ""

if uploaded_img:
    pil_img = Image.open(uploaded_img)
    st.image(pil_img, caption="আপলোডকৃত ছবি", width=250)
    
    if st.button("ছবি থেকে প্রশ্ন বের করুন"):
        if not openai_api_key:
            st.error("দয়া করে সাইডবারে OpenAI API Key দিন।")
        else:
            with st.spinner("OpenAI Vision দিয়ে ছবি পড়া হচ্ছে..."):
                extracted_image_question = extract_question_from_image(pil_img, openai_api_key)
                st.info(f"📷 **ছবি থেকে সংগৃহীত প্রশ্ন:** {extracted_image_question}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("আপনার প্রশ্নটি এখানে লিখুন...")
query = user_input or (extracted_image_question if uploaded_img else None)

if query:
    if not openai_api_key:
        st.error("দয়া করে সাইডবারে OpenAI API Key প্রদান করুন।")
    elif st.session_state.rag_pipeline is None:
        st.warning("দয়া করে সাইডবার থেকে প্রথমে একটি PDF প্রসেস করুন।")
    else:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("PDF থেকে উত্তর খোঁজা হচ্ছে..."):
            response = st.session_state.rag_pipeline(query, st.session_state.chat_history)
            answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)
            
            if "context" in response and len(response["context"]) > 0:
                top_doc = response["context"][0]
                matched_page = top_doc.metadata.get("page", 0)
                
                diagrams = extract_diagrams_from_page(st.session_state.pdf_path, matched_page)
                if diagrams:
                    st.markdown("**🖼️ সংশ্লিষ্ট পেজ থেকে এক্সট্র্যাক্ট করা ডায়গ্রাম:**")
                    for d_img in diagrams:
                        st.image(d_img, width=400)
            
            pdf_out = generate_simple_pdf(answer)
            st.download_button(
                label="📥 উত্তরটি PDF হিসেবে ডাউনলোড করুন",
                data=pdf_out,
                file_name="AI_Response.pdf",
                mime="application/pdf"
            )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=answer)
        ])
    
