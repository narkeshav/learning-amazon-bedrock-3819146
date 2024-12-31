import boto3
import json
import logging
import time
from langchain_aws import ChatBedrock
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import BedrockEmbeddings


print("App is running with latest changes")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure streamlit app
st.set_page_config(page_title="Social Media Training Bot", page_icon="📖")

# Custom title and subtitle
st.markdown("<h1 style='text-align: center;'>📖 Social Media Training Bot</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>by Keshav</h3>", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "waiting_for_answer" not in st.session_state:
    st.session_state.waiting_for_answer = False
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = {}

# Add clear conversation button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    StreamlitChatMessageHistory(key="langchain_messages").clear()
    st.rerun()

# Customization - In the sidebar
st.sidebar.title("Model Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 64, 1024, 512, 64)

# Rate limiting parameters
CALLS_PER_MINUTE = 5
SECONDS_PER_MINUTE = 60

def rate_limited_call(func, *args, **kwargs):
    current_time = time.time()
    if current_time - st.session_state.last_request_time < SECONDS_PER_MINUTE / CALLS_PER_MINUTE:
        time_to_wait = (SECONDS_PER_MINUTE / CALLS_PER_MINUTE) - (current_time - st.session_state.last_request_time)
        time.sleep(time_to_wait)
    st.session_state.last_request_time = time.time()
    return func(*args, **kwargs)

# Define convenience functions
# Update the config_llm function
@st.cache_resource
def config_llm(temperature: float, top_p: float, max_tokens: int) -> ChatBedrock:
    client = boto3.client('bedrock-runtime')
    model_kwargs = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    llm = ChatBedrock(model_id=model_id, client=client, model_kwargs=model_kwargs)
    return llm

@st.cache_resource
def config_vector_db(filename: str) -> FAISS:
    """
    Configure and return a FAISS vector store.

    Args:
        filename (str): The name of the PDF file to be loaded.

    Returns:
        FAISS: A configured FAISS vector store.
    """
    client = boto3.client('bedrock-runtime')
    embeddings = BedrockEmbeddings(
        client=client,
        model_id="amazon.titan-embed-text-v2:0"
    )
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    vectorstore_faiss = FAISS.from_documents(pages, embeddings)
    return vectorstore_faiss

# Configuring the llm and vector store
llm = config_llm(temperature, top_p, max_tokens)
vectorstore_faiss = config_vector_db("03_06b/social-media-training.pdf")

logger.info("Application started")
logger.info(f"LLM configured with temperature: {temperature}, max_tokens: {max_tokens}")
logger.info(f"Vector database configured with file: social-media-training.pdf")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")
    st.session_state.messages.append({"role": "assistant", "content": "How can I help you?"})

# Creating the template   
my_template = ChatPromptTemplate.from_messages([
    ("system", "You are a conversational assistant designed to help answer questions from an employee. "
               "You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. "
               "Do not say things like 'according to the training or handbook or based on or according to the information provided...'.\n\n"
               "<Information>\n{info}\n</Information>"),
    ("human", "{input}")
])

# Create llm chain
question_chain = LLMChain(
    llm=llm,
    prompt=my_template,
    output_key="answer"
)

def retrieve_relevant_documents(query: str) -> str:
    """
    Retrieve relevant documents based on the given query.

    Args:
        query (str): The user's input query.

    Returns:
        str: A string containing the content of relevant documents.
    """
    docs = vectorstore_faiss.similarity_search_with_score(query)
    info = ""
    for doc in docs:
        info += doc[0].page_content + '\n'
    return info

def generate_response(query: str, info: str) -> dict:
    """
    Generate a response using the LLM based on the query and relevant information.

    Args:
        query (str): The user's input query.
        info (str): Relevant information retrieved from the vector database.

    Returns:
        dict: A dictionary containing the LLM's response.
    """
    return rate_limited_call(question_chain.invoke, {"input": query, "info": info})

def update_chat_history(query: str, response: str):
    """
    Update the chat history with the user's query and the assistant's response.

    Args:
        query (str): The user's input query.
        response (str): The assistant's response.
    """
    msgs.add_user_message(query)
    msgs.add_ai_message(response)
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response})

def simulate_progress():
    """
    Simulate a progress bar for visual feedback.
    """
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    progress_bar.empty()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is your question?"):
    if not prompt.strip():  # Check if the prompt is empty or just whitespace
        st.warning("Please enter a valid question.")
    else:
        # Store the prompt in session state
        st.session_state.current_prompt = prompt
        st.session_state.waiting_for_answer = True
        # Don't append to messages yet
        st.rerun()

# Generate and display response
if st.session_state.waiting_for_answer:
    query = st.session_state.get('current_prompt', '')
    # Display the user's message once
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving relevant information..."):
                simulate_progress()
                info = retrieve_relevant_documents(query)
            with st.spinner("Generating response..."):
                simulate_progress()
                output = generate_response(query, info)
            response = output['answer']
            # Only update chat history after response is generated
            update_chat_history(query, response)
            st.markdown(response)
            
           
            logger.info(f"Received query: {query}")
            logger.info(f"Retrieved relevant documents")
            logger.info(f"Generated response: {response[:50]}...")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            st.error("I'm sorry, but I encountered an error while processing your request. Please try again.")
        finally:
            st.session_state.waiting_for_answer = False
            st.session_state.current_prompt = None


            #Feedback Mechanism
            # Define columns for feedback buttons
            # Simplified feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                    st.success("Thank you for your helpful feedback!")
                    logger.info(f"Feedback: Helpful for query: {query[:30]}...")

            with col2:
                if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                    st.error("We're sorry the response wasn't helpful. We'll work on improving it.")
                    logger.info(f"Feedback: Not Helpful for query: {query[:30]}...")

            
    st.rerun()
