# pip install langchain
# pip install -qU langchain-groq
# pip install langchain_community

import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, trim_messages, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory


load_dotenv()

model = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# O problema abaixo é que o modelo não tem memória, então não consegue responder uma 
# pergunta com base no histórico de conversa
chamada = model.invoke([HumanMessage(content="Hello im fernando")])
print(chamada.content)
chamada = model.invoke([HumanMessage(content="What's my name?")])
print(chamada.content)

# Para resolver o problema de cima, é necessário passar o histórico de mensagens
chamada = model.invoke(
    [
        HumanMessage(content="Hello im fernando"),
        AIMessage(content="Hola Fernando! It's nice to meet you. Is there something I can help you with or would you like to chat?"),
        HumanMessage(content="What's my name?"),
    ]
)
print(chamada.content)

# Outra forma de resolver o problema é usando o RunnableWithMessageHistory, que permite
# gerenciar o histórico de mensagens de forma mais eficiente.

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "abc2"}} # configuração para gerenciar o histórico de mensagens

chamada = with_message_history.invoke(
    [HumanMessage(content="Hello im fernando")],
    config=config,
)
print(chamada.content) # Hola Fernando! Nice to meet you! How's it going?

chamada = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print(chamada.content) # Your name is Fernando!

config = {"configurable": {"session_id": "abc3"}} # nova sessão, novo histórico de mensagens
chamada = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print(chamada.content) # I'm happy to help! However, I'm a large language model, I don't have the
                       # ability to remember or know your name unless you've told me before. If 
                       # you'd like to share your name with me, I'd be happy to learn it and use 
                       # it in our conversation!

# O histórico da outra sessão continua salvo, então é só voltar a usar o id de antes que é possível acessar
config = {"configurable": {"session_id": "abc2"}}

chamada = with_message_history.invoke(  
    [HumanMessage(content="What's my name?")],
    config=config,
)

print(chamada.content) # Your name is Fernando!

prompt = ChatPromptTemplate.from_messages( # Template de como o modelo deve se comportar
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model # corrente

chamada = chain.invoke({"messages": [HumanMessage(content="Hello im fernando")]})

print(chamada.content) # Hello Fernando! It's nice to meet you. I'm here to help you
                       # with any questions or topics you'd like to discuss. What's on your mind today?

with_message_history = RunnableWithMessageHistory(chain, get_session_history) # tambem é possível usar o historico das mensgaens
config = {"configurable": {"session_id": "abc5"}} # nova sessão

chamada = with_message_history.invoke(
    [HumanMessage(content="Hello im fernando")],
    config=config,
)

print(chamada.content) # Hello Fernando! It's nice to meet you. I'm here to help you with any questions or topics you'd like to discuss. What's on your mind today?

chamada = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
print(chamada.content) # Your name is Fernando! How can I assist you further today?

prompt = ChatPromptTemplate.from_messages( # Adicionou variavel ao prompt
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | model

chamada = chain.invoke(
    {"messages": [HumanMessage(content="Hello im fernando")],"language": "portuguese"} # Agora passa a variavel tambem
)
print(chamada.content) # Olá Fernando! É um prazer conhecê-lo.

trimmer = trim_messages(
    max_tokens=65, # historico limitado a 65 toknes
    strategy="last", # mantem as ultimas mensagens, se necessario, remove as mais antigas
    token_counter=model, 
    include_system=True, # inclui mensagens do sistema (caso tenha)
    allow_partial=False, # não permite cortar mensagens no meio
    start_on="human", 
)

messages = [ # mensagens de exemplo
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hello im fernando"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like chocolate ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="yes?"),
    AIMessage(content="yes!"),
]

chamada = trimmer.invoke(messages)

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

chamada = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "Portuguese",
    }
)
print(chamada.content) # Fernando!

# stream serve apenas para ficar mais amigavel ao usuário
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    [HumanMessage(content="Hello, im fernando. tell me a joke")],
    config=config,
):
    print(r.content, end="|")