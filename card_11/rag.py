from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vectordb import url_to_vector
from main import number_of_oscars

prompt = ChatPromptTemplate.from_template(""" Responda a pergunta com base no contexto fornecido. Se não souber a resposta, diga que não sabe.
{context}
Pergunta: {input}
""")

document_chain = create_stuff_documents_chain(number_of_oscars, prompt)
retriever = url_to_vector('https://pt.wikipedia.org/wiki/Oppenheimer_(filme)')
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "Quantos oscars o filme Oppenheimer ganhou no ano de 2024?"})
print(response['output'])