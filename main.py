from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from flask import session

load_dotenv()

def initialize_session_state():
  # create a session object  

  if 'history' not in session:
     session['history'] = []

  if 'generated' not in session:
     session['generated'] = ["Hello! Ask me anything about"]

  if 'past' not in session:
      session['past'] = ["Hey!"]

def conversation_chat(query, chain, history):
  result = chain({"question": query, "chat_history": history})
  history.append((query, result["answer"]))
  return result["answer"]


def create_conversational_chain(vector_store):
    load_dotenv()
    llm = Replicate(
    streaming=True,
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    callbacks=[StreamingStdOutCallbackHandler()],
    input={"temperature": 0.01, "max_length": 500, "top_p": 1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    return chain

app = FastAPI()

def display_chat_history(chain, prompt):
  # reply_container = st.container()
  # container = st.container()

  # with container:
  #     with st.form(key='my_form', clear_on_submit=True):
  #         user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
  #         submit_button = st.form_submit_button(label='Send')

      # if submit_button and user_input:
          # with st.spinner('Generating response...'):
              output = conversation_chat(prompt, chain, session['history'])
              session['past'].append(prompt)
              session['generated'].append(output)
  
# if session['generated']:
#       with reply_container:
#           for i in range(len(st.session_state['generated'])):
#               message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
#               message(st.session_state["generated"][i], key=str(i))

@app.get("/")
async def read_root():
    return {"Hello": "World"}
  
@app.get("/items/{item_id}")
async def read_item(item_id: str):
  # Open the file and read its content
  with open('AIRole_Pharmacy.txt', 'r') as file:
      content = file.read()
  # Print the content of the file
  print(content)

  prompt = item_id
  initialize_session_state()

  text = []

  text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
  text_chunks = text_splitter.split_documents(text)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
  vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
  chain = create_conversational_chain(vector_store)
  display_chat_history(chain, prompt)
  
  return {content}


# @app.get("/items/{item")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}