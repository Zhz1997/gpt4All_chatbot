import lancedb
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import GPT4AllEmbeddings

db = lancedb.connect("./lancedb")

table = db.open_table("data")

vectorStore = LanceDB(table, GPT4AllEmbeddings())

localPath = ("./models/wizardlm-13b-v1.2.q4_0.gguf")

template = """
Use the following pieces of context and chat history to answer the questions.
Context: {context}
---
Chat history: {chat_history}
---
Question: {question}
"""


prompt = PromptTemplate(
    template = template,
    input_variables=["context, question", "chat_history"]
)

callbacks = [StreamingStdOutCallbackHandler()]

llm = GPT4All(
    model=localPath,
    callbacks=callbacks,
    verbose = True
)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    vectorStore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": prompt}
)

chat_history = []
print("=======================")
print("Type 'exit' to stop,")
while True:
    query = input("please enter your question: ")
    if query.lower() == 'exit':
        break
    result = qa({"question": query, "chat_history": chat_history})
    print("\n")