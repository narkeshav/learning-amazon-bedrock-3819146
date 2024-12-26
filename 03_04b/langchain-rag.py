# Import libraries
import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Define vectorstore
global vectorstore_faiss

# Define convenience functions
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
vectorstore_faiss = config_vector_db("03_04e/social-media-training.pdf")

# Creating the template   
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a conversational assistant designed to help answer questions from an employee. "
               "You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. "
               "Do not say things like 'according to the training or handbook or based on or according to the information provided...'.\n\n"
               "Context: {context}"),
    ("human", "{input}")
])

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
retrieval_chain = create_retrieval_chain(vectorstore_faiss.as_retriever(), document_chain)

# Get question, perform similarity search, invoke model and return result
while True:
    question = input("\nAsk a question about the social media training manual:\n")

    # invoke the model, providing additional context
    response = retrieval_chain.invoke({"input": question})

    # display the result
    print(response["answer"])
    