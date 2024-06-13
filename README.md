# Streamlit App for Policy Inspector
This Streamlit application serves as a tool to generate responses based on user input, utilizing the Llama-2-13b model from Hugging Face and a pre-trained MiniLM-L6-v2 model for embeddings. The application allows users to upload a dataset in Excel format containing 'Query' and 'Response' columns. Upon uploading the dataset, users can input a prompt, and the system generates a response based on the input prompt using the pre-trained models.

## Features
- Dataset Upload: Users can upload an Excel file containing a dataset with 'Query' and 'Response' columns.
- Prompt-based Response: Users can input a prompt, and the system generates a response based on the input prompt using the pre-trained models.
- Response Exploration: The application provides the capability to explore the raw response object and the source text.

1. Clone the repository:

```bash
git clone https://github.com/KishanMunjpara2002/Policy-Inspector.git
```

3. Run the Streamlit application:

```bash
streamlit run application copy.py
```

## Usage

1.After running the Streamlit app, the user interface will open in your default web browser.
2.Upload your dataset in Excel format using the provided file uploader. The dataset should contain 'Query' and 'Response' columns.
3.Once the dataset is uploaded successfully, you will see the first few rows of the dataset.
4.Input your prompt in the text input box provided.
5.Press Enter, and the application will generate a response based on the input prompt.
6.The generated response will be displayed on the screen.
7.To explore the raw response object and the source text, you can expand the respective sections.

## Requirements

- Python 3.7 or higher
- Streamlit
- pandas
- transformers
- torch
- llama-index
- langchain

## Configuration

Before running the application, make sure to set up the required configurations in the  file. 

- `name`: The name of the Hugging Face model.
- `auth_token`: Authentication token for accessing the model.
- `quantization_config`: Configuration for quantization.
- `system_prompt`: System prompt for the language model.
- `query_wrapper_prompt`: Query wrapper prompt for the language model.
- `context_window`: Context window size for the language model.
- `max_new_tokens`: Maximum number of new tokens for the language model.
- `rope_scaling`: Rope scaling configuration for the language model.
- `chunk_size`: Chunk size for the service context.
- `model_name`: Name of the Hugging Face model for embeddings.

## Libraries and Frameworks Used
Streamlit: Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
Pandas: Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation library built on top of the Python programming language.
Hugging Face Transformers: Transformers provides thousands of pre-trained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation, etc., in over 100 languages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


