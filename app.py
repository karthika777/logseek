import os
import gradio as gr
from azure.storage.blob import BlobServiceClient
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from guardrails import Guard
from guardrails.hub import ProfanityFree
import pickle
import time
from io import BytesIO
import pandas as pd

def pickles_from_blob(connection_string, container_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
       
        blobs_list = container_client.list_blobs()
        
        data_frames = []
        
        for blob in blobs_list:
            if blob.name.endswith('.pkl'):  
                blob_client = container_client.get_blob_client(blob.name)
                
                blob_stream = BytesIO()
                blob_client.download_blob().readinto(blob_stream)
                
                blob_stream.seek(0)
          
                df = pickle.load(blob_stream)
                data_frames.append(df)

        if data_frames:
            concatenated_df = pd.concat(data_frames, ignore_index=True)
        else:
            print("No pickle files found in the container.")
            concatenated_df = pd.DataFrame()

    except Exception as e:
        print(f"An error occurred: {e}")
        concatenated_df = pd.DataFrame() 

    finally:
        container_client.close()
        blob_service_client.close()

    return concatenated_df

connection_string = "DefaultEndpointsProtocol=https;AccountName=aimlloganalyticstest;AccountKey=ikPneEuYonwekpcNhuWK5bUHqr3Cc2jt4IgI0vX29PQqbAPeUN5UVsiCGrJXI6+7cB0ccL+durBq+ASt9LxRDQ==;EndpointSuffix=core.windows.net"
container_name = "pickle-files"

df_final = pickles_from_blob(connection_string, container_name)

Settings.llm = AzureOpenAI(
    engine="gpt-35-turbo",
    model="gpt-35-turbo",
    temperature=0.0,
    azure_endpoint="https://testopenaiforrag123.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview",
    api_key="1b60b8b6bdd8474381342caf30f0af14",
    api_version="2023-07-01-preview"
)
query_engine = PandasQueryEngine(df=df_final, synthesize_response=True, response_mode="compact")
guard = Guard().use(ProfanityFree())

chat_history = []

class PersonaAgent:
    def __init__(self, persona):
        self.persona = persona

    def handle_query(self, query):
        identity_queries = ["who is this?", "who are you?", "what is your name?"]
        help_queries = ["how can you help me?", "what can you do?", "what is your purpose?", "how can I use you?"]
        greetings_message = ["hi", "hello", "hey", "howdy", "greetings"]
        log_queries = [
            "what is the total number of log entries in the dataset?", 
            "how is this number distributed across different Operation Names?",
            "what are the Operation Names in the dataset?"
        ]

        if query.lower() in identity_queries:
            response = f"I am {self.persona['name']}, {self.persona['role']}. I am here to assist you with any questions or issues you may have on Azure logs😄."
        elif query.lower() in greetings_message:
            response = f"Hello! I am {self.persona['name']}, {self.persona['role']}. I am here to assist you with any questions or issues you may have on Azure logs😄."
        elif query.lower() in help_queries:
            response = (
                "I can assist you with a variety of tasks, including answering questions, "
                "providing information, and helping with specific tasks based on your needs. "
                "Just let me know what you need help with, and I'll do my best to assist you."
            )
        elif any(q in query.lower() for q in log_queries):
            response = "I can provide information on log entries and their distribution. Please specify the dataset or log file you're referring to."
        else:
            return None
        
        return response

persona = {
    "name": "LogSeek - AI",
    "role": "your personal log assistant bot"
}

persona_agent = PersonaAgent(persona)

def my_llm_api(prompt: str = None, **kwargs) -> str:
    if prompt:
        chat_history.append({"role": "user", "content": prompt})
        
        full_prompt = f"User: {prompt}\n"
        for chat in chat_history:
            full_prompt += f"{chat['role']}: {chat['content']}\n"
        
        response = f"Generated response for: {prompt}"  
        
        chat_history.append({"role": "assistant", "content": response})
        return response

    return "Please provide a prompt."

def guard(func, prompt):
    response = func(prompt=prompt)
    validated_output = response.strip() 
    return type("GuardRailsResponse", (object,), {"validated_output": validated_output})

def query_model(prompt):
    persona_response = persona_agent.handle_query(prompt)
    if persona_response:
        output = persona_response
    else:
        response = query_engine.query(prompt).response
        output = response
    
    return output

def gradio_interface(query, history):
    output = query_model(query)
    words = output.split()
    ans = ""
    for token in words:
        ans += " " + token
        time.sleep(0.1)
        yield ans

interface = gr.Interface(
    fn=gradio_interface,  
    inputs=["text", "state"], 
    outputs=["text"],  
    title="LogSeek - AI", 
    description="Your personal log assistant bot."
)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))  
    interface.launch(server_name="0.0.0.0", server_port=port)
