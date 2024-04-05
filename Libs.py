import os
import boto3, json
from dotenv import load_dotenv
from botocore.client import Config
from langchain.llms.bedrock import Bedrock
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from botocore.client import Config
from langchain.chains import RetrievalQA
from langchain_community.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate


load_dotenv()

def call_claude_sonet_stream(input):

    prompt = """Dịch sang tiếng Việt với phong cách Hán Việt 
    Ví dụ bên dưới:
    <example>
        全服第一个通关（仅限原服完成）黄昏圣殿·空副本  => Người đầu tiên của toàn server vượt Phó Bản (Chỉ giới hạn server cũ hoàn thành) Điện Hoàng Hôn-Không
        全服第一个通关（仅限原服完成）覆霜城·空副本  => Người đầu tiên của toàn server vượt Phó Bản (Chỉ giới hạn server cũ hoàn thành) Thành Sương Mù-Không
        全服第一个通关（仅限原服完成）众魂之境团队副本  => Người đầu tiên của toàn server vượt Phó Bản Quân Đoàn (Chỉ giới hạn server cũ hoàn thành) Chúng Hồn Cảnh
    </example>
    Content to translate: """ + str(input) + " => "
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0, 
        "top_k": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    bedrock = boto3.client(service_name="bedrock-runtime")  
    response = bedrock.invoke_model_with_response_stream(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    stream = response['body']
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                 delta = json.loads(chunk.get('bytes').decode()).get("delta")
                 if delta:
                     yield delta.get("text")    

def search(input_text): 
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="JAHBTIXPHK",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 1}}
    )
    model_kwargs_claude = {"temperature": 0, "top_k": 0, "max_tokens_to_sample": 100}
    llm = Bedrock(model_id="anthropic.claude-v2", model_kwargs=model_kwargs_claude)

    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    return qa(input_text)