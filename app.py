import chainlit as cl 
import PyPDF2
import docx
from docx import Document
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.schema import Document
from chromadb.config import Settings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import sys
from typing import Optional
import logging   
import time
import random
import pyttsx3
from typing import List
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.base import VectorStore
from prompt import EXAMPLE_PROMPT, BASIC_PROMPT, TRANS_PROMPT, PROMPT, FILE_QUERY, BASIC_QUERY, TRANSLATOR, WELCOMINGS
from languages import gpt, llama, cohere
from langchain.llms import Cohere
from langchain.llms import Together


backoff_in_seconds = float(os.getenv("BACKOFF_IN_SECONDS", 20))
max_retries = int(os.getenv("MAX_RETRIES", 10))

logging.basicConfig(stream = sys.stdout,
                    format = '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def backoff(attempt : int) -> float:
    return backoff_in_seconds * 2**attempt + random.uniform(0, 1)


def get_text(file):

    file_stream = BytesIO(file.content)
    extension = file.name.split('.')[-1]

    text = ''

    if extension == "pdf":
        reader = PyPDF2.PdfReader(file_stream)
        for i in range(len(reader.pages)):
            text +=  reader.pages[i].extract_text()
    elif extension == "docx":
        doc = docx.Document(file_stream)
        paragraph_list = []
        for paragraph in doc.paragraphs:
            paragraph_list.append(paragraph.text)
        text = '\n'.join(paragraph_list)
    elif extension == "txt":
        text = file_stream.read().decode('utf-8')
    
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, #ensure full sentence
        length_function=len 
    )

    chunks = text_splitter.split_text(text) #list of chunks
    return chunks 

def get_vstore(
    *, docs: List[Document], embeddings: Embeddings, metadatas
) -> VectorStore:
    # Initialize Chromadb client to enable resetting and disable telemtry
    client = chromadb.EphemeralClient()
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False,
        persist_directory=".chromadb",
        allow_reset=True,
    )

    # Reset the vstore, prevent old files confusing
    search_engine = Chroma(client=client, client_settings=client_settings)
    search_engine._client.reset()

    search_engine = Chroma.from_texts(
        client=client,
        texts = docs,
        embedding=embeddings,
        client_settings=client_settings,
        metadatas = metadatas
    )

    return search_engine

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key="answer"
    )

    llm = cl.user_session.get("llm")
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs = {"prompt": PROMPT, "document_prompt": EXAMPLE_PROMPT},
        memory = memory,
    )
    return chain

def get_translation_chain(vectorstore):
    
    llm = cl.user_session.get("llm")
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs = {"prompt": TRANS_PROMPT},
        verbose = True,
    )
    return chain

def get_basic_chain():
    prompt = BASIC_PROMPT
    llm = cl.user_session.get("llm")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = LLMChain(
        llm = llm, 
        prompt=prompt,
        verbose=False,
        memory=memory)
    cl.user_session.set("chain",chain)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

#continuously on a loop
@cl.on_message
async def main(message: cl.Message):
    chat_type = cl.user_session.get("actions")
    chain = cl.user_session.get("chain") 

    if chat_type == "basic":
        for attempt in range(max_retries):
            try:
                res = await chain.acall(message.content, 
                                callbacks=[cl.AsyncLangchainCallbackHandler()])
                break 
            except Exception:
                wait_time = backoff(attempt)
                logger.exception(f"Rate limit reached. Waiting {wait_time} seconds and trying again")
                time.sleep(wait_time)
                break

        await cl.Message(content=res["text"], 
                    author="Chatbot").send()
        answer = res["text"]
        #speak_text(answer)

    elif chat_type == "translate":
        if message.content == "list":
            await cl.Message(content=cl.user_session.get("lg"), 
                        author="Chatbot").send()
        
        else:
            res = await chain.acall(message.content, 
                                    callbacks=[cl.AsyncLangchainCallbackHandler()])
            await cl.Message(content=res["answer"], 
                        author="Chatbot").send()
            answer = res["answer"]

    elif chat_type == "file":
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        for attempt in range(max_retries):
            try:
                res = await chain.acall(message.content,
                                        callbacks = [cb])
                break
            except Exception:
                wait_time = backoff(attempt)
                logger.exception(f"Rate limit reached. Waiting {wait_time} seconds and trying again")
                time.sleep(wait_time)
                break

        answer = res["answer"]
        sources = res["sources"].strip()
        source_elements = []

        metadatas = cl.user_session.get("metadatas")
        all_sources = [m["source"] for m in metadatas]
        texts = cl.user_session.get("texts")

        if sources != "":
            found_sources = []

            # Add the sources to the message
            for source in sources.split(","):
                source_name = source.strip().replace(".", "")
                # Get the index of the source
                try:
                    index = all_sources.index(source_name)
                except ValueError:
                    continue
                text = texts[index]
                found_sources.append(source_name)
                # Create the text element referenced in the message
                source_elements.append(cl.Text(content=text, name=source_name))

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo source is found. "
        
        if cb.has_streamed_final_answer:
            cb.final_stream.elements = source_elements
            await cb.final_stream.update()
        else:
            await cl.Message(content=answer, 
                    elements=source_elements, 
                    author="Chatbot").send()
            
            #speak_text(answer)
    else:
        await cl.Message(content="Please select one of the option to start!",
                         author="Chatbot").send()
        
# see user wants basic or file query
@cl.action_callback(name="confirm")
async def action_callback(action: cl.Action):
    
    value = action.value

    actions = cl.user_session.get("actions")
    if actions:
        for action in actions:
            await action.remove()
        cl.user_session.set("actions", None)

    if value == '1':
        cl.user_session.set("actions", "basic")
        chain = get_basic_chain()
        await cl.Message(content=BASIC_QUERY,
                         author="Chatbot").send()
    
    elif value == '2' or value == '0':
        if value == '2':
            content = TRANSLATOR
            cl.user_session.set("actions", "translate")

        elif value == '0':
            content = FILE_QUERY
            cl.user_session.set("actions", "file")
        
        files = None  
        while files == None:
            files = await cl.AskFileMessage(
                content = content, 
                accept=["application/pdf",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "text/plain"],
                author = "Chatbot",
                max_size_mb = 20,
                timeout = 86400,
                raise_on_timeout = False
            ).send()

        file = files[0]

        msg = cl.Message(
            content=f"Processing '{file.name}' ...",
            author = "Chatbot",
        )
        await msg.send()

        text = get_text(file)

        text_chunks = get_text_chunks(text)

        metadatas = [{"source": f"pg-{i}"} for i in range(len(text_chunks))]

        #vectorstore = await get_vectorstore(text_chunks, metadatas)

        embeddings = OpenAIEmbeddings()
        #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
        vstore = await cl.make_async(get_vstore)(
            docs = text_chunks, 
            embeddings = embeddings, 
            metadatas = metadatas
        )

        if value == '2':
            chain = get_translation_chain(vstore)
            msg.content = f"Processing '{file.name}' done.\nYou can tell me the language you want. \nOr say 'list' to see the available language"
            await msg.update()
        
        elif value == '0':
            chain = get_conversation_chain(vstore)
            msg.content = f"Processing '{file.name}' done.\nFeel free to ask any question!"
            await msg.update()

        cl.user_session.set("metadatas", metadatas)
        cl.user_session.set("texts", text_chunks)
        cl.user_session.set("chain",chain)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="GPT",
            markdown_description="The underlying LLM model is **GPT-3.5-TURBO**.",
            icon="https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg",
        ),
        cl.ChatProfile(
            name="LLAMA-2",
            markdown_description="The underlying LLM model is **LLAMA-2**.",
            icon="https://icons.iconarchive.com/icons/iconarchive/incognito-animal-2/256/Lama-icon.png",
        ),
        cl.ChatProfile(
            name="COMMAND",
            markdown_description="The underlying LLM model is **COMMAND**.",
            icon="https://asset.brandfetch.io/idfDTLvPCK/idfkFVkJdH.png",
        ),
    ]
    
# when a new chat is start
@cl.on_chat_start
async def start():
    chat = cl.user_session.get("chat_profile")
    cl.user_session.set("memory",None)

    if chat == "GPT":
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature = 0.5,
            openai_api_key=os.environ["OPENAI_API_KEY"]
            )
        cl.user_session.set("lg",gpt)
    elif chat == "LLAMA-2":
        llm = Together(
            model="togethercomputer/llama-2-70b-chat",
            temperature = 0.5,
            together_api_key=os.environ["TOGETHER_API_KEY"]
            )
        cl.user_session.set("lg",llama)
    elif chat == "COMMAND":
        llm = Cohere(
            model="command",
            temperature = 0.5,
            cohere_api_key=os.environ["COHERE_API_KEY"]
            )
        cl.user_session.set("lg",cohere)

    cl.user_session.set("llm",llm)
    cl.user_session.get("user")

    await cl.Avatar(
        name="Chatbot",
        path="assets/chatbot.jpeg"
    ).send()

    await cl.Avatar(
        name="admin",
        path="assets/user.jpg"
    ).send()

    
    actions = [
        cl.Action(name="confirm",label="1. Documents Q&A 📚",value="0",description="Ask about your files!"),
        cl.Action(name="confirm",label="2. Translate documents A ↔️ 文", value="2",description="Translate your files!"),
        cl.Action(name="confirm",label="3. Basic Chat 💬", value="1", description="Have a basic chat!")
        ]

    cl.user_session.set("actions", actions)

    msg = cl.Message(content=WELCOMINGS, 
            actions=actions,
            author="Chatbot",
            )
    
    await msg.send()
    

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
    

