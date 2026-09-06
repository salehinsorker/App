import tempfile
import io
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from fpdf import FPDF
import google.generativeai as genai
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
except ModuleNotFoundError:
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Custom Gemini Native Embeddings Wrapper ---
class CustomGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        self.api_key = api_key.strip()
        self.model_name = model_name
        genai.configure(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            res = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(res['embedding'])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        res = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return res['embedding']

# --- Custom Bulletproof Hybrid Retriever ---
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
st.set_page_config(page_title="Free Gemini Multimodal RAG", page_icon="✨", layout="wide")
st.title("✨ Free Multimodal Hybrid RAG Agent (Powered by Gemini API)")

# --- Session State Initialization ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
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

# --- Helper 3: Vision Question Extraction ---
def extract_question_from_image(pil_image, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model_names = ['models/gemini-1.5-flash', 'gemini-1.5-flash-latest', 'models/gemini-2.0-flash']
    
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([
                "Extract the exact question, problem, or main subject from this image clearly into plain text.",
                pil_image
            ])
            return response.text
        except Exception:
            continue
            
    return "ছবি থেকে টেক্সট বের করা সম্ভব হয়নি। দয়া করে ম্যানুয়ালি প্রশ্নটি টাইপ করুন।"

# --- Sidebar Setup ---
with st.sidebar:
    st.header("🔑 ১. ফ্রি API Key ও PDF আপলোড")
    gemini_api_key = st.text_input("Google Gemini API Key দিন", type="password")

    uploaded_pdf = st.file_uploader("PDF ফাইল আপলোড করুন", type=["pdf"])
    
    if uploaded_pdf and gemini_api_key and st.button("PDF প্রসেস করুন"):
        with st.spinner("PDF প্রসেস করা হচ্ছে..."):
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
                clean_api_key = gemini_api_key.strip()
                embeddings = None
                errors_log = []
                
                candidate_models = ["models/text-embedding-004", "models/embedding-001"]
                
                for m_name in candidate_models:
                    try:
                        emb_test = CustomGeminiEmbeddings(api_key=clean_api_key, model_name=m_name)
                        emb_test.embed_query("test query")
                        embeddings = emb_test
                        break
                    except Exception as err:
                        errors_log.append(f"{m_name}: {str(err)}")

                if embeddings is None:
                    st.error("❌ কোনো এম্বেডিং মডেল দিয়ে সংযোগ স্থাপন করা যায়নি। বিস্তারিত এরর:")
                    for err_msg in errors_log:
                        st.caption(f"• {err_msg}")
                    st.warning("👉 দয়া করে [Google AI Studio](https://aistudio.google.com/) থেকে একটি নতুন API Key তৈরি করে চেষ্টা করুন।")
                else:
                    vectorstore = FAISS.from_documents(cleaned_splits, embeddings)
                    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                    bm25_retriever = BM25Retriever.from_documents(cleaned_splits)
                    bm25_retriever.k = 3

                    hybrid_retriever = CustomHybridRetriever(retrievers=[bm25_retriever, vector_retriever])

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash", 
                        google_api_key=clean_api_key,
                        temperature=0
                    )

                    context_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Given a chat history and the latest user question, rephrase it to be a standalone question."),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                    history_aware_retriever = create_history_aware_retriever(llm, hybrid_retriever, context_prompt)

                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Answer the question using ONLY the provided context below. If context doesn't contain the answer, say 'তথ্যটি PDF-এ পাওয়া যায়নি।'\n\n{context}"),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ])
                    
                    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
                    st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
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
        if not gemini_api_key:
            st.error("দয়া করে সাইডবারে Gemini API Key দিন।")
        else:
            with st.spinner("Gemini Vision দিয়ে ছবি পড়া হচ্ছে..."):
                extracted_image_question = extract_question_from_image(pil_img, gemini_api_key)
                st.info(f"📷 **ছবি থেকে সংগৃহীত প্রশ্ন:** {extracted_image_question}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("আপনার প্রশ্নটি এখানে লিখুন...")
query = user_input or (extracted_image_question if uploaded_img else None)

if query:
    if not gemini_api_key:
        st.error("দয়া করে সাইডবারে Gemini API Key প্রদান করুন।")
    elif st.session_state.rag_chain is None:
        st.warning("দয়া করে সাইডবার থেকে প্রথমে একটি PDF প্রসেস করুন।")
    else:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("PDF থেকে উত্তর খোঁজা হচ্ছে..."):
            response = st.session_state.rag_chain.invoke({
                "input": query,
                "chat_history": st.session_state.chat_history
            })
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
        
