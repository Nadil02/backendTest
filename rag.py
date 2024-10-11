import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import os
import shutil

load_dotenv()

genai.configure(api_key=os.environ["API_KEY"])

llm = genai.GenerativeModel("gemini-1.5-flash")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

DATA_PATH = "data"
CHROMA_PATH = "chroma"

DATA_PATH_ = "uploads"
CHROMA_PATH_ = "chromaforpdf"


def load_document(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True,
    )
    chunks = text_splitter.split_documents(documents=documents)
    print(f"Split {len(documents)} page documents into {len(chunks)} chunks.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, embeddings_model, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

def save_to_chroma_(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH_):
        shutil.rmtree(CHROMA_PATH_)

    db = Chroma.from_documents(
        chunks, embeddings_model, persist_directory=CHROMA_PATH_
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH_}")

def create_knowledge_base():
    documents = load_document(DATA_PATH)
    chunks = split_text(documents=documents)
    save_to_chroma(chunks)

def create_knowledge_base_():
    documents = load_document(DATA_PATH_ )
    chunks = split_text(documents=documents)
    save_to_chroma_(chunks)


def getAnswer(query, candidate, field, name):
    print(name)
    candidates_to_exclude = [
    "Anura Kumara Dissanayake (NPP candidate)",
    "Sajith Premadasa (SJP candidate)",
    "Ranil Wickramasinghe",
    "Namal Rajapakshe (SLPP candidate)"]

    if candidate not in candidates_to_exclude:
        #create_knowledge_base_()
        results = vector_db_.similarity_search_with_relevance_scores(query, k=15)
    else:
        results = vector_db.similarity_search_with_relevance_scores(query, k=15)


    # if name is not "Anura Kumara Dissanayake (NPP candidate)" or "Sajith Premadasa (SJP candidate)" or "Ranil Wickramasinghe" or "Namal Rajapakshe (SLPP candidate)":
    #     create_knowledge_base_()
    #     results = vector_db_.similarity_search_with_relevance_scores(query, k=15)
    # else:
    #     results = vector_db.similarity_search_with_relevance_scores(query, k=15)

    if(len(results)==0):
        answer = "No results !"

    context_with_metadata = []
    for doc, score in results:
            metadata = doc.metadata
            page_content = doc.page_content
            context_with_metadata.append(f"Metadata: {metadata}\nContent: {page_content}\n\n---\n\n")
        
    context_text = "\n\n".join(context_with_metadata)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context_text, candidate=candidate, field=field, name=name)

    answer = llm.generate_content(prompt)

    return answer.text

def getSummary(summary_query):
     answer = llm.generate_content(summary_query)
     return answer.text


PROMPT_TEMPLATE = """
Answer the question using only the following context:

{context}

Based on the above context and using the metadata PDF name, extract the goals for the candidate and field provided below:

Candidate (party): {candidate}
Field: {field}
Candidate short name: {name}

Use the metadata PDF name to identify the relevant candidate for each text chunk. Ensure that the extracted information pertains specifically to the given candidate and field. Do not include goals or information relevant to other candidates.

If the provided context does not have specific information on the field for the relevant candidate, try to extract relevant information based on the context of the relevent candidate.
If no relevant information can be extracted, provide a single list item: "Unable to find {field} goals from {name}'s manifesto."

Provide the answer in the following format:
- Use bullet points (HTML <ul> tags) for clarity.
- Limit the points to a maximum of 6.
- Include all relevant details from the context.
- Ensure the topic relates to the mini description of the topic (e.g., school education for the education field).

Answer template:
<ul>
    <li>topic_of_the_point_1 - mini_description</li>
    <li>topic_of_the_point_2 - mini_description</li>
    ...
</ul>
"""




vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)
vector_db_ = Chroma(persist_directory=CHROMA_PATH_, embedding_function=embeddings_model)