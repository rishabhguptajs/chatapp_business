import sys
import constants

from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

file_path = (
    "./salaries.csv"
)

query = sys.argv[1]

loader = CSVLoader(file_path=file_path)

embedding_model = OpenAIEmbeddings(api_key=constants.OPENAI_KEY)

index_create = VectorstoreIndexCreator(embedding=embedding_model)
index = index_create.from_loaders([loader])

print(index.query(query, llm=ChatOpenAI))