# custom_Chatbot
This project implements a custom chatbot that can answer user queries based on the content of uploaded documents. It leverages several powerful tools, including LangChain, Hugging Face Hub, TextSplitter, and Streamlit, to provide a seamless and interactive user experience.

Overview
The chatbot is built to handle document-based queries by reading uploaded PDF files, processing their contents, and allowing users to ask questions related to the document's content. The core components of the chatbot include LangChain for natural language processing, HuggingFaceHub for integrating powerful language models, TextSplitter to handle large documents by breaking them into manageable chunks, and Streamlit to create an interactive web application for user interaction.

Key Features:
Document Upload: Users can upload PDF documents that the chatbot will read and process.
Chunking: Large documents are divided into smaller sections using the TextSplitter to improve the chatbot's ability to find relevant information efficiently.
Natural Language Processing: The chatbot is powered by models from Hugging Face Hub integrated through LangChain, which enables accurate and context-aware responses to user queries.
Streamlit Interface: The entire chatbot application is hosted on a simple and user-friendly web interface using Streamlit, allowing users to interact with the chatbot through their browsers.
Workflow:
The user uploads a PDF document via the Streamlit interface.
The document is processed using TextSplitter to break it into smaller chunks for better query matching.
A question is posed by the user, and the chatbot uses LangChain to convert the query into vector embeddings.
The processed document data is searched for relevant information using similarity search from LangChain's vectorstore.
A Hugging Face model retrieves and generates a contextually accurate response based on the user's query and the document's content.
The chatbot displays the answer on the Streamlit interface.
