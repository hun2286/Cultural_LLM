import os
import re
import fitz
import socket
import torch
from dotenv import load_dotenv
from typing import Optional, Union, TypedDict
from konlpy.tag import Komoran

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# LangGraph 0.6
from langgraph.graph import StateGraph, END

komoran = Komoran()

# 1. LLM 로드 함수
def load_hf_causal_lm_pipeline(
        repo_id: str, model_dir: Optional[Union[str, os.PathLike]] = None,
        task: str = "text-generation", max_new_tokens: int = 2048,
        temperature: float = 0.3, top_p: float = 0.9, top_k: int = 20,
        repetition_penalty: float = 1.2, torch_dtype=torch.bfloat16,
        expose_prompt: bool = False
    ):
    
    hostname = socket.gethostname()
    if hostname == 'ubuntu' and model_dir is None:
        model_dir = "/volume/hf_cache/hub"

    model = AutoModelForCausalLM.from_pretrained(repo_id, cache_dir=model_dir, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        return_full_text=expose_prompt
    )
    return HuggingFacePipeline(pipeline=pipe)

# 2. 환경 변수 및 기본 설정
load_dotenv()

pdf_folder = r"/volume/bgr_storage/embedding_data/유산별 보고서/무형유산"
persist_dir = "./Intangible_db"

print("Gemma-3 로딩 중...")
llm = load_hf_causal_lm_pipeline(repo_id="unsloth/gemma-3-12b-it-bnb-4bit")
print("Gemma-3 초기화 완료!")

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# 3. PDF 처리 함수
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',,+', ',', text)
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    text = re.sub(r'\s*,\s*\n', '\n', text)
    return text.strip()

def pdf_to_markdown(pdf_path):
    md_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                t = page.get_text("text")
                if t.strip():
                    md_text += clean_text(t) + "\n\n"
        return md_text
    except Exception:
        print(f"[오류] PDF 변환 실패 : {pdf_path}")
        return ""

def load_pdf_safe(pdf_path):
    md_text = pdf_to_markdown(pdf_path)
    if not md_text.strip():
        return None
    return [Document(page_content=md_text, metadata={"source": os.path.splitext(os.path.basename(pdf_path))[0]})]

def load_all_pdfs_recursive(root_folder):
    all_docs, failed = [], []
    for dp, _, files in os.walk(root_folder):
        pdfs = [f for f in files if f.lower().endswith(".pdf")]
        for pdf_file in pdfs:
            path = os.path.join(dp, pdf_file)
            try:
                docs = load_pdf_safe(path)
                if not docs:
                    failed.append(path)
                else:
                    all_docs.extend(docs)
                    print(f"{path} 처리 완료")
            except Exception as e:
                print(f"{path} 오류: {e}")
                failed.append(path)

    print(f"\n총 Document 수: {len(all_docs)}")
    return all_docs

# 4. Vector DB 생성/로드
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("DB 새로 생성합니다...")
    docs = load_all_pdfs_recursive(pdf_folder)
    if docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        split_docs = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(split_docs, embedding=embedding_model, persist_directory=persist_dir)
        vectorstore.persist()
        print("임베딩 완료!")
    else:
        vectorstore = None
else:
    print("기존 DB 사용")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}) if vectorstore else None

# 5. RAG core function
def extract_keywords(question):
    nouns = komoran.nouns(question)
    return [w for w in nouns if len(w) > 1]

def title_matches_keywords(title, keywords):
    return all(kw.lower() in title.lower() for kw in keywords)

def build_prompt(context, question):
    return f"""
<start_of_turn>system
당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
- 문서 내용만 활용하며, 문서에 없는 내용은 '정보 없음'이라고 답하세요.
- 각 항목은 제목과 내용 한 줄, 빈 줄 순서로 작성하며, 제목은 'OO 소개' 또는 'OO 개요' 형태로 형용사 없이 작성하세요.
- 문단 사이 불필요한 쉼표를 제거하고, 문장 끝은 마침표 또는 줄바꿈으로 마무리하세요.

- 문서 내용이 충분할 경우, 단순 요약을 넘어 다음 요소들을 포함하여 내용을 더 풍부하게 작성하세요:
  • 등장 배경이나 시대적 맥락  
  • 핵심 개념의 의미  
  • 문서 속 인물, 사건, 사례에 대한 구체적 설명  
  • 기술·문화적 특징 또는 변화 과정  
  • 문서 흐름을 기반으로 한 자연스러운 부연 설명  

- 단, 문서에 근거 없는 가정, 해석, 창작, 추론은 절대 포함하지 말고, 반드시 문서에 실린 내용만 확장하여 설명하세요.
- 문서에 정보가 적어 내용 확장이 불가능한 경우에는 문서 내용을 정확히 전달하는 것을 우선하세요.

- 답변 마지막에 참고한 PDF 출처를 [출처: PDF 제목] 형태로 모아서 표기하세요. 
- 본문에는 PDF 제목, 보고서 이름, '참고:', '참고문헌' 등의 표현을 포함하지 마세요.
- '삭제', '누락', '삭제됨', '삭제 요청', '설명', '다음과 같습니다' 등의 안내 문장은 포함하지 마세요.
- 첫 줄에 답변과 관련없는 문장은 포함하지 마세요.
<end_of_turn>
<start_of_turn>user
문서 내용:
{context}

질문:
{question}
<end_of_turn>

<start_of_turn>model
""".strip()

def rag_core(question):
    if not retriever:
        return "DB 없음"

    docs = retriever.invoke(question)
    if not isinstance(docs, list):
        docs = [docs]

    keywords = extract_keywords(question)
    filtered = [d for d in docs if title_matches_keywords(d.metadata.get("source", ""), keywords)]

    if not filtered:
        filtered = docs

    context = "\n\n".join(d.page_content.strip() for d in filtered)
    prompt = build_prompt(context, question)
    return llm.invoke(prompt)

# 6. LangGraph 0.6 StateGraph 구성
class RagState(TypedDict):
    question: str
    answer: str

def node_retrieve_and_generate(state: RagState):
    q = state["question"]
    answer = rag_core(q)
    return {"answer": answer}

graph = StateGraph(RagState)

graph.add_node("rag", node_retrieve_and_generate)
graph.set_entry_point("rag")
graph.add_edge("rag", END)

app = graph.compile()

# 7. 실행
if __name__ == "__main__":
    print("=== LangGraph 0.6 기반 RAG 실행 ===")
    while True:
        query = input("\n질문 입력 (exit 종료): ").strip()
        if query.lower() == "exit":
            break

        result = app.invoke({"question": query})
        print("\n[답변]:\n", result["answer"])
        print("=" * 60)
