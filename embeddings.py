import json
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from pprint import pprint
import os

from sqlalchemy.testing.plugin.plugin_base import config

from config import OPENAI_API_KEY, OPENAI_PROXY, LANGCHAIN_API_KEY

def test1():
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
    text = 'Благодаря появлению LLM возникла новая професссия будущего. \
    Это промпт-инженер, который наилучшим образом направляет LLM на правильный путь . \
    Этот специалист должен обладать творческим мышлением и аналитическими способностями, \
    и желательно знанием методов машинного обучения и NLP и навыками программирования'

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,#250
        chunk_overlap=0,#100
        separators = ['. ']#
    )
    splited_data = splitter.split_text(text)
    pprint(splited_data)

def telegram_get_text_from_message(message) -> str:
    if isinstance(message, list):
        res = []
        for message_ in message:
            if isinstance(message_, str):
                res_ = [message_]
            elif isinstance(message_, dict):
                res_ = []
                for key, value in message_.items():
                    if key == 'text':
                        res_.append(value)
                    elif key == 'type':
                        pass
                    elif key == 'collapsed':
                        pass
                    elif key == 'document_id':
                        pass
                    elif key == 'href':
                        res_.append(value)
                    else:
                        print(f'passed key: {key}')
            else:
                print(f'passed message_: {message_}')
                res_ = [message_]
            res.append(' '.join(res_))
        return ' '.join(res)
    elif isinstance(message, str):
        return message
    else:
        print(f'passed message: {message}')
    return message

def telegram_extract_text_from_chat_json(chat_json: str, result_file: str) -> None:

    with open(chat_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    text_chunks = []
    for message in data['messages']:
        if message['type'] == 'message' and message['text']:
            txt = telegram_get_text_from_message(message['text'])
            if 'from' in message:
                text_chunks.append(f'date: {message['date']} user: {message["from"]} text: {txt}')
            else:
                text_chunks.append(f'date: {message["date"]} text: {txt}')
        # print(message)

    # pprint(text_chunks)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(text_chunks, f, ensure_ascii=False, indent=4)

def create_openai_embeddings(persist_directory: str, chuncks_json_file: str) -> None:
    # persist_directory = ''
    # embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY) if OPENAI_PROXY is None or OPENAI_PROXY == "" \
    #     else OpenAIEmbeddings(api_key=OPENAI_API_KEY, http_client=httpx.Client(proxy=OPENAI_PROXY))
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) if not OPENAI_PROXY \
           else OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_PROXY)
    with open(chuncks_json_file, 'r', encoding='utf-8') as f:
        splitted_texts = json.load(f)

    # vectordb = Chroma.from_documents(
    #     documents=splitted_texts,
    #     embedding=embedding,
    #     persist_directory=persist_directory
    # )
    vectordb = Chroma.from_texts(
        texts=splitted_texts,
        embedding=embedding,
        persist_directory=persist_directory
    )

    print(f'Ebbeddings created. Amount: {vectordb._collection.count()}')

def check_chroma_db(persist_directory: str) -> None:
    # embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) if not OPENAI_PROXY \
    #        else OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_PROXY)
    vectordb = Chroma(
        # embedding_function=embedding,
        persist_directory=persist_directory
    )

    # Fetch all documents
    # documents = vectordb.get()

    collection = vectordb._collection
    count = collection.count()
    print(f'Size of collection: {count}')

    items = collection.peek(limit=1)
    print(f'Size of ebbedding: {len(items['embeddings'][0])}')
    print('First 1 items: ')
    pprint(items)

def test_embeddings(persist_directory: str, question: str) -> None:
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) if not OPENAI_PROXY \
           else OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_PROXY)
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    # answers = vectordb.similarity_search(question, k=20)
    # for answer in answers:
    #     print(answer.page_content)
    #     # print(answer.metadata)  # .source
    #     print('-----')

    results_with_scores = vectordb.similarity_search_with_score(question, k=20)
    for idx, (doc, score) in enumerate(results_with_scores):
        print(f"Result {idx+1}:")
        print("Content:", doc.page_content)
        print("Score:", score)
        # print("Metadata:", doc.metadata)
        print()

def ask_openai_with_embeddings(persist_directory: str, question: str, k: int = 4) -> None:
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) if not OPENAI_PROXY \
           else OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, openai_proxy=OPENAI_PROXY)
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

    retriever = vectordb.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\nФрагмент текста\n\n".join(doc.page_content for doc in docs)

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        openai_api_key=OPENAI_API_KEY,
        openai_proxy=OPENAI_PROXY
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print(answer)

def detect_file_encoding(fname: str) -> None:
    import chardet

    # Step 1: Read a sample of the file to detect encoding
    with open(fname, 'rb') as file:
        raw_data = file.read(10000)  # Read the first 10,000 bytes (or adjust as needed)

    # Step 2: Detect encoding
    result = chardet.detect(raw_data)
    encoding = result['encoding']

    print(f'Detected encoding: {encoding}')



if __name__ == "__main__":
    # step 1
    # telegram_extract_text_from_chat_json('chat_pe.json', 'chat_pe_extracted.json')

    # step 2
    # create_openai_embeddings('chroma_db','chat_pe_extracted.json')

    # step 3 - optional
    # check_chroma_db('chroma_db')

    # step 4
    # test_embeddings('chroma_db', 'Какие нейросети обсуждались?')
    # test_embeddings('chroma_db', 'Нейросети')
    # test_embeddings('chroma_db', 'llm')
    # test_embeddings('chroma_db', 'Любое обсуждение нейросетей: chatgpt, claude, gemini и т.п.')

    # step 5
    ask_openai_with_embeddings('chroma_db', 'Какие нейросети обсуждались?')

    # test1()
    # test2()
    # detect_encoding('chat_pe1.json')
