
#  Document Q&A Assistant (RAG)

A web-based **AI-powered Document Question & Answer Assistant** built using **Retrieval-Augmented Generation (RAG)**.  
Users can upload documents and ask questions that are answered **strictly based on the uploaded content**, with clear citations.



##  Features

- Upload documents in **PDF, DOCX, TXT, Markdown**
- Automatic text extraction and chunking
- **Token-based chunking (500–1000 tokens with overlap)**
- Semantic search using **FAISS**
- Answers generated **only from retrieved document content**
- **Citations included** (document name + page number)
- Clear fallback when information is missing
- Simple and clean **chat interface**
- Options to **clear chat** and **reset knowledge base**

---

##  Architecture Overview


##  Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **LLM & Embeddings**: OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)  
- **Vector Database**: FAISS  
- **Tokenization**: tiktoken  
- **Document Parsing**: pypdf, python-docx  

## Installation & Setup

**1 Clone the repository**

git clone https://github.com/PrathamShukla3102/your-repo-name.git

cd your-repo-name

**2 Create virtual environment (optional but recommended)**

python -m venv venv

source venv/bin/activate   # Windows: venv\Scripts\activate

**3 Install dependencies**

pip install -r requirements.txt

**4 Set environment variables**

add api key in .env file:

OPENAI_API_KEY=your_openai_api_key

**5 Run the Application**

streamlit run app.py

The app will open in your browser.

## How to Use

Upload one or more documents from the sidebar

-Click Build / Rebuild Index

-Ask questions in the chat box

-View answers with citations

-Use Clear Chat or Reset Knowledge Base as needed

## Example Questions

-What is the objective of this document?

-Summarize the key findings.

-What technologies are mentioned?
**If the information is not present:I couldn’t find this information in the uploaded document.**




