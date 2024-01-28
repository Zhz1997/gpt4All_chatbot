import lancedb
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

db = lancedb.connect("./lancedb")
table = db.create_table(
    "data",
    data=[
        {
            "vector": GPT4AllEmbeddings().embed_query("hello world"),
            "text": "hello world",
            "id": "1",
        }
    ],
    mode="overwrite",
)

splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)

rawDoc = TextLoader("./docs/FakeWiki.txt").load()

documents = splitter.split_documents(rawDoc)

vectorStore = LanceDB(connection=table, embedding=GPT4AllEmbeddings())

vectorStore.add_documents(documents=documents)
