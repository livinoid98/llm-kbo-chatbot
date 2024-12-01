import os
import openai
from dotenv import load_dotenv
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from datasets import load_dataset
from ragas import EvaluationDataset

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

request_data = [
    {
        "user_input": "2024년 KBO 리그 정규 시즌 우승 팀은 어디인가요?",
        "reference": "2024년 KBO 리그 정규 시즌 우승 팀은 KIA TIGERS입니다.",
        "retrieved_contexts": ["KIA TIGERS는 2024년 KBO 리그에서 정규 시즌 우승을 차지했습니다."],
        "response": "KIA TIGERS가 2024년 KBO 리그 정규 시즌 우승을 차지했습니다."
    },
    {
        "user_input": "2024년 KBO 리그에서 최다 세이브를 기록한 투수는 누구인가요?",
        "reference": "2024년 KBO 리그에서 최다 세이브를 기록한 투수는 고우석입니다.",
        "retrieved_contexts": ["고우석은 2024년 KBO 리그에서 최다 세이브를 기록했습니다."],
        "response": "2024년 KBO 리그 최다 세이브 투수는 고우석입니다."
    },
    {
        "user_input": "2024년 KBO 리그에서 한 경기 최다 탈삼진을 기록한 투수는 누구인가요?",
        "reference": "2024년 KBO 리그에서 한 경기 최다 탈삼진을 기록한 투수는 안우진입니다.",
        "retrieved_contexts": ["안우진은 2024년 한 경기에서 최다 탈삼진을 기록한 투수입니다."],
        "response": "2024년 KBO 리그에서 한 경기 최다 탈삼진을 기록한 투수는 안우진입니다."
    }
]

eval_dataset = EvaluationDataset.from_list(request_data)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    FactualCorrectness(llm=evaluator_llm), 
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]

results = evaluate(dataset=eval_dataset, metrics=metrics)

df = results.to_pandas()
print(df.head())