from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_data(csv_path):
    return CSVLoader(csv_path, encoding="utf-8").load()


def persist(model_name, documents, output_directory):
    embedding = HuggingFaceEmbeddings(
        model_name=model_name
    )
    Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=output_directory,
    )


def main():
    model_name = "intfloat/multilingual-e5-base"
    csv_path = "datas/categorized_cpes.csv"
    output_directory = "vectorstore"

    documents = load_data(csv_path=csv_path)
    persist(model_name, documents=documents, output_directory=output_directory)

    print("Vectorstore saved.")


if __name__ == "__main__":
    main()
