#Import libraries
import boto3
#from langchain_community.embeddings import BedrockEmbeddings
#from langchain_community.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_aws import ChatBedrock, BedrockEmbeddings  


#Define vectorstore
global vectorstore_faiss


#Define convenience functions
def config_llm():
    client = boto3.client('bedrock-runtime')
    
    model_kwargs = { 
        "max_tokens_to_sample": 512,
        "temperature":0.1,  
        "top_p":1
    }  

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    # llm = Bedrock(model_id=model_id, client=client)
    

    # New instantiation  
    llm = ChatBedrock(model_id=model_id, client=client)  
    llm.model_kwargs = model_kwargs
    #bedrock_embeddings = BedrockEmbeddings(client=client)     #unnecessary

    return llm

def config_vector_db(filename):
    client = boto3.client('bedrock-runtime')
    embedding_model_id = "anthropic.claude-3-sonnet-20240229-v1:0" 
    bedrock_embeddings = BedrockEmbeddings(model_id=embedding_model_id, client=client)
    #bedrock_embeddings = BedrockEmbeddings(client=client)
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()

    # Ensure correct input formatting for embeddings  
    # texts = [page.page_content for page in pages]  
    #formatted_texts = [{"prompt": text, "max_tokens_to_sample": 512} for text in texts]

    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss


def vector_search (query):
    docs = vectorstore_faiss.similarity_search_with_score(query)
    info = ""
    for doc in docs:
        info+= doc[0].page_content+'\n'
    return info    


#Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db("03_04e/social-media-training.pdf")

#Creating the template   
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from an employee. 
    You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Only answer the question. Do not say things like "according to the training or handbook or according to the information provided...".
    
    <Information>
    {info}
    </Information>
    

    {input}

Assistant:
"""

#Configure prompt template
prompt_template = PromptTemplate(
    input_variables=['input', 'info'],
    template= my_template
)

#Create llm chain
question_chain = LLMChain(
    llm = llm,
    prompt = prompt_template,
    output_key = "answer"
)

#Get question, peform similarity search, invoke model and return result
while True:
    question = input ("\nAsk a question about the social media training manual:\n")

    #perform a similarity search
    info = vector_search(question)

    #invoke the model, providing additional context
    output = question_chain.invoke({'input' : question,  'info' : info})

    #display the result
    print(output['answer'])




# from typing import List
  
# import boto3  
# import json  
# import random
# import time
# import logging  
# from typing import List  
# from langchain_aws import ChatBedrock, BedrockEmbeddings  
# from langchain.prompts.prompt import PromptTemplate  
# from langchain_community.document_loaders import PyPDFLoader  
# from langchain_community.vectorstores import FAISS  
# from langchain.chains import LLMChain  
  
# # Define vectorstore  
# global vectorstore_faiss  
  
# # Helper function for exponential backoff  
# def exponential_backoff(retries, max_delay=60):  
#     delay = min(max_delay, (2 ** retries) + random.uniform(0, 1))  
#     time.sleep(delay)  
  
# # Messages API wrapper for invoking model  
# def invoke_model_with_messages_api(client, model_id, input_text):  
#     input_body = {  
#         "messages": [  
#             {"role": "user", "content": "You are a helpful assistant."},  
#             {"role": "assistant", "content": input_text}  
#         ],  
#         "max_tokens": 512,  
#         "anthropic_version": "bedrock-2023-05-31",  
#     }  
#     body = json.dumps(input_body)  
      
#     retries = 0  
#     max_retries = 5  
      
#     while retries < max_retries:  
#         try:  
#             response = client.invoke_model(  
#                 body=body,  
#                 modelId=model_id,  
#                 accept="application/json",  
#                 contentType="application/json",  
#             )  
#             response_body = json.loads(response.get("body").read())  
#             return response_body["content"]  
#         except client.exceptions.ThrottlingException as e:  
#             logging.warning(f"ThrottlingException: {e}, retrying...")  
#             retries += 1  
#             exponential_backoff(retries)  
#         except Exception as e:  
#             logging.error(f"Error raised by inference endpoint: {e}")  
#             raise e  
  
#     raise Exception("Max retries exceeded")  
  
# # Define convenience functions  
# def config_llm():  
#     client = boto3.client('bedrock-runtime')  
#     model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Replace with your valid model ID  
  
#     # Create a wrapper for ChatBedrock to use the Messages API  
#     class ChatBedrockWrapper(ChatBedrock):  
#         def _call(self, prompt, stop=None):  
#             return invoke_model_with_messages_api(client, model_id, prompt)  
  
#     # Instantiate the wrapped ChatBedrock  
#     llm = ChatBedrockWrapper(model_id=model_id, client=client)  
#     return llm  
  
# def config_vector_db(filename):  
#     client = boto3.client('bedrock-runtime')  
#     embedding_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Replace with your valid embedding model ID  
  
#     # Wrapper for BedrockEmbeddings to use Messages API  
#     class BedrockEmbeddingsWrapper(BedrockEmbeddings):  
#         def _embedding_func(self, text: str) -> List[float]:  
#             input_body = {  
#                 "messages": [  
#                     {"role": "user", "content": "You are a helpful assistant."},  
#                     {"role": "assistant", "content": text}  
#                 ],  
#                 "max_tokens": 512,  
#                 "anthropic_version": "bedrock-2023-05-31",  
#             }  
#             body = json.dumps(input_body)  
              
#             retries = 0  
#             max_retries = 5  
              
#             while retries < max_retries:  
#                 try:  
#                     response = client.invoke_model(  
#                         body=body,  
#                         modelId=embedding_model_id,  
#                         accept="application/json",  
#                         contentType="application/json",  
#                     )  
#                     response_body = json.loads(response.get("body").read())  
#                     return response_body["content"]  
#                 except client.exceptions.ThrottlingException as e:  
#                     logging.warning(f"ThrottlingException: {e}, retrying...")  
#                     retries += 1  
#                     exponential_backoff(retries)  
#                 except Exception as e:  
#                     logging.error(f"Error raised by inference endpoint: {e}")  
#                     raise e  
  
#             raise Exception("Max retries exceeded")  
  
#     bedrock_embeddings = BedrockEmbeddingsWrapper(model_id=embedding_model_id, client=client)  
#     loader = PyPDFLoader(filename)  
#     pages = loader.load_and_split()  
  
#     vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)  
#     return vectorstore_faiss  
  
# def vector_search(query):  
#     docs = vectorstore_faiss.similarity_search_with_score(query)  
#     info = ""  
#     for doc in docs:  
#         info += doc[0].page_content + '\n'  
#     return info  
  
# # Configuring the llm and vector store  
# llm = config_llm()  
# vectorstore_faiss = config_vector_db("03_04e/social-media-training.pdf")  
  
# # Creating the template  
# my_template = """  
# Human: You are a conversational assistant designed to help answer questions from an employee.  
# You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Only answer the question. Do not say things like "according to the training or handbook or according to the information provided...".  
# <Information>  
# {info}  
# </Information>  
# {input}  
# Assistant:  
# """  
  
# # Configure prompt template  
# prompt_template = PromptTemplate(  
#     input_variables=['input', 'info'],  
#     template=my_template  
# )  
  
# # Create llm chain  
# question_chain = LLMChain(  
#     llm=llm,  
#     prompt=prompt_template,  
#     output_key="answer"  
# )  
  
# # Get question, perform similarity search, invoke model and return result  
# while True:  
#     question = input("\nAsk a question about the social media training manual:\n")  
#     # Perform a similarity search  
#     info = vector_search(question)  
#     # Invoke the model, providing additional context  
#     output = question_chain.invoke({'input': question, 'info': info})  
#     # Display the result  
#     print(output['answer'])  