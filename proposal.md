We want to merge Amazon review dataset into our project.

First, we can select Electronics category from Amazon review 23 dataset.

Then use a small portion of it to generate mock user profile.

Then use a small portion of this category to generate mock product. 

Project Description:

[Functions and Users] This project plans to leverage publicly available Amazon product reviews and metadata from UCSD to develop a new web-based application as an intelligent shopping decision-making assistance platform. The major functions include: (1) building an efficient ETL (Extract, Transform, Load) pipeline for large-scale data import and indexing; (2) utilizing LLM (Large Language Model) to understand complex and vague semantic queries (e.g., "safe and educational outdoor toys suitable for a 5-year-old boy"); and (3) using RAG (Retrieval-Augmented Generation) mechanisms to recommend specific products to users based on product reviews and frequently searched tags. The target users are online shoppers who need to make quick and accurate purchasing decisions from vast amounts of reviews, and data analysts who want to summarize and analyze market feedback for specific product categories.

[Significance] Existing e-commerce search functions often rely on simple keyword matching, making it difficult to handle complex semantic queries. Users also struggle to extract relevant information from thousands of product reviews. This tool uses AI technology to reduce the decision-making burden on consumers and improve the efficiency of information retrieval, directly addressing a pain point faced by millions of online shoppers daily.

[Approach] This is a standalone web-based tool. Our core technical approach uses embeddings and vector databases (such as Milvus, Pinecone, or SQLite-vss) for semantic search, connected to LLMs (Large Language Models) to understand queries and generate natural language recommendations. We will leverage LangChain for chaining calls to embeddings, LLMs, and database interfaces, Pandas for handling raw JSON data from Amazon, and Streamlit for the front-end interface. The primary data resource is the Amazon Reviews Dataset from UCSD (https://jmcauley.ucsd.edu/data/amazon/). The main risk is that the dataset is massive and embedding computations are expensive. To mitigate this, we will start with a specific subset (e.g., "Electronics" or "Books") for initial development and testing before scaling up.

[Evaluation] We will write scripts to verify that the original JSON data is correctly imported into the vector database with lossless embedding. We will also compare the relevance of returned results between traditional keyword search and AI semantic search to demonstrate the usefulness and correctness of our implementation.

[Timeline] Weeks 1–2: Download a subset of the dataset, analyze the data structure, and design the vector database schema. Weeks 3–4 (Milestone 1): Complete ETL script development, preprocess the dataset subset, compute embeddings, and import data into the vector database. Weeks 5–6 (Milestone 2): Develop core backend logic (semantic search + LLM-based recommendation generation) with LangChain, and build a basic Streamlit interface. Week 7: Performance optimization, final documentation writing, and project demonstration.

