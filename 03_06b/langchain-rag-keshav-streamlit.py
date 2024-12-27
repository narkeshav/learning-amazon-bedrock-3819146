import boto3
import json
from langchain_aws import ChatBedrock
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import BedrockEmbeddings


# Configure streamlit app
st.set_page_config(page_title="Social Media Training Bot", page_icon="📖")
st.title("📖 Social Media Training Bot by Keshav")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "waiting_for_answer" not in st.session_state:
    st.session_state.waiting_for_answer = False

# Add clear conversation button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    StreamlitChatMessageHistory(key="langchain_messages").clear()
    st.rerun()

# Define convenience functions
@st.cache_resource
def config_llm():
    client = boto3.client('bedrock-runtime')
    model_kwargs = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.1,
        "top_p": 1
    }
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    llm = ChatBedrock(model_id=model_id, client=client, model_kwargs=model_kwargs)
    return llm

@st.cache_resource
def config_vector_db(filename):
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
llm = config_llm()
vectorstore_faiss = config_vector_db("03_06b/social-media-training.pdf")

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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is your question?"):
    if not prompt.strip():  # Check if the prompt is empty or just whitespace
        st.warning("Please enter a valid question.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.waiting_for_answer = True
        st.rerun()

# Generate and display response
if st.session_state.waiting_for_answer:
    with st.chat_message("user"):
        st.markdown(st.session_state.messages[-1]["content"])
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # retrieve relevant documents using a similarity search
            docs = vectorstore_faiss.similarity_search_with_score(st.session_state.messages[-1]["content"])
            info = ""
            for doc in docs:
                info += doc[0].page_content + '\n'

            # invoke llm
            output = question_chain.invoke({"input": st.session_state.messages[-1]["content"], "info": info})

            # adding messages to history
            msgs.add_user_message(st.session_state.messages[-1]["content"])
            msgs.add_ai_message(output['answer'])

            # display the output
            st.markdown(output['answer'])
            st.session_state.messages.append({"role": "assistant", "content": output['answer']})
    
    st.session_state.waiting_for_answer = False
    st.rerun()