#Policy-Inspector

Got it! I've added the README file to your GitHub repository. If you need any further modifications or have any other questions, feel free to ask!
## Installation

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

Before running the application, make sure to set up the required configurations in the `app.py` file. 

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

## Contributors

- [Your Name](https://github.com/YourGitHubUsername)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

You can include this `README.md` file in your GitHub repository for users to understand how to set up and use your Policy Inspector Streamlit application. Let me know if you need any further assistance!
