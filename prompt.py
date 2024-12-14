from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

template = """Use the following pieces of context to answer the users question.
    If you don't know the answer, just say you don't know, never try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    The "SOURCES" part should be a reference to the source of the document from which you got your answer.

    Example of your response should be:

    ```
    The answer is foo
    SOURCES: xyz
    ```

    Begin!
    ===========
    {summaries}
    ===========

    Given the following is the older conversation between you and user:
    {chat_history}


    Answer the following question, if the question is a simple greeting, or thank you, or sorry, no need to return a "SOURCES" part in your answer.
    User:[INST]{question}[/INST]
    """

trans_template = """You are going to translate the following context into language specified in the question.
=========
{summaries}
=========
QUESTION: [INST]{question}[/INST]

FINAL ANSWER:"""

TRANS_PROMPT = PromptTemplate(
    template=trans_template,input_variables=["summaries","question"]
)

BASIC_PROMPT = ChatPromptTemplate(
        messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("[INST]{question}[/INST]"),]
    ) 

PROMPT = PromptTemplate(
    template=template, input_variables=["summaries","chat_history","question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)


FILE_QUERY = """\
1. Upload a .pdf, .txt, or .docx file
2. Ask any question about the file!
"""

BASIC_QUERY = """\
Hi! Feel free to ask any questions~
"""

TRANSLATOR = """\
1. Upload a .pdf, .txt, or .docx file
2. Tell the target language!
"""

WELCOMINGS = """\
👋 Hello there! Welcome to Pandai Chat~
We are delighted to have you here. Pandai Chat is your intelligent companion designed to assist you effortlessly. Whether you seek engaging conversation or require assistance with file-related tasks, our chatbot is here to cater to your needs.
\nTo get started, kindly choose from the following options: 
"""

