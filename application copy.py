import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core import VectorStoreIndex, Document
# Create centered main title
st.title('👷🏻 Pocily Inspector')

# Define variable to hold llama2 weights naming
name = "meta-llama/Llama-2-13b-chat-hf"
auth_token = " "

@st.cache_resource
def get_tokenizer_model():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='D:/Python Environment for Sales/Stremlit_App_For/model_7b/', use_auth_token=auth_token)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=0.8,
        llm_int8_has_fp16_weights=True,
        bnb_4bit_use_double_quant=True,
    )
    
    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        name, 
        cache_dir='D:/Python Environment for Sales/Stremlit_App_For/model_13b/',
        use_auth_token=auth_token, 
        torch_dtype=torch.float16, 
        rope_scaling={"type": "dynamic", "factor": 2}, 
        quantization_config=quantization_config
    )
    
    # Create a system prompt 
    system_prompt = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as 
    helpfully as possible, while being safe. Your answers should not include
    any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain 
    why instead of answering something not correct. If you don't know the answer 
    to a question, please don't share false information.
    Your goal is to provide answers relating to the Policy of company.<</SYS>>
    """
    
    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")
    
    # Create a HF LLM using the llama index wrapper 
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=512,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        model=model,
        tokenizer=tokenizer
    )

    # Create and dl embeddings instance  
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Create new service context instance
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embeddings
    )
    
    # Set the service context
    set_global_service_context(service_context)

    return tokenizer, model, embeddings

tokenizer, model, embeddings = get_tokenizer_model()

uploaded_file = st.file_uploader("Upload dataset", type=["xlsx"])

if not uploaded_file:
    st.write("No file uploaded. Please upload an Excel file to continue.")
else:
    try:
        # Load the dataset
        df = pd.read_excel(uploaded_file)
        
        st.write("Dataset loaded successfully:")
        st.write(df.head())

        # Debug: Print out the column names to check if 'Query' and 'Response' are present
        st.write("Columns in the dataset:")
        st.write(df.columns.tolist())

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Check for required columns
        if 'query' not in df.columns or 'response' not in df.columns:
            st.error("The dataset must contain 'Query' and 'Response' columns.")
        else:
            # Extract queries and responses
            queries = df['query'].tolist()
            responses = df['response'].tolist()

            # Create Document instances for indexing
            documents = [Document(text=response, doc_id=str(i)) for i, response in enumerate(responses)]

            # Create an index from the documents
            index = VectorStoreIndex.from_documents(documents)

            # Setup index query engine using LLM 
            query_engine = index.as_query_engine()

            # Create a text input box for the user
            prompt = st.text_input('Input your prompt here')

            # If the user hits enter
            if prompt:
                response = query_engine.query(prompt)
                # ...and write it out to the screen
                st.write(response.response)

                # Display raw response object
                with st.expander('Response Object'):
                    st.write(response.response)
                # Display source text
                with st.expander('Source Text'):
                    st.write(response.get_formatted_sources())
    except Exception as e:
        st.error(f"An error occurred: {e}")

