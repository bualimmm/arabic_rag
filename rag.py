import os
import re
import ssl
import streamlit as st

import httpx
import pandas as pd
from langchain_community.vectorstores import FAISS
from loguru import logger

from embeddings import CustomEmbeddings
from settings import settings

system_prompt_qa = """مهمتك هي تقديم إجابات دقيقة وموجزة بناءً على سياق معطى مستند "بيان الميزانية العامة للسعودية للعام المالي 2024م". ستتلقى سؤالاً وسياقًا يتألف من 10 مقاطع مستخرجة من المستند مرتبة حسب مدى صلتها بالسؤال (قد لا تكون المقاطع مرتبطة ببعضها البعض بالضرورة)..
خطوات العمل:
1. تحليل السؤال:
   * الغرض الرئيسي من السؤال: (اذكر هنا الغرض الرئيسي من السؤال - طلب معلومة، تعريف، مقارنة، إلخ)
   * الكيانات الرئيسية في السؤال: (اذكر هنا الكيانات الرئيسية - أسماء، أماكن، تواريخ، أرقام، إلخ)
2. تحليل السياق:
   * اقرأ كل مقطع من السياق بعناية لتحديد مدى صلته بالسؤال.
   * ابحث عن تطابق مباشر أو غير مباشر بين الكلمات الرئيسية والكيانات في السؤال والمقاطع السياقية.
   * قيّم ما إذا كانت المعلومات الموجودة في السياق كافية للإجابة على جميع جوانب السؤال.
3. تحديد مدى توافر المعلومات في السياق:
   * الغرض الرئيسي: (حدد ما إذا كان السياق يعالج الغرض الرئيسي للسؤال بشكل مباشر، جزئي، أو غير موجود)
   * الكيانات الرئيسية: (بالنسبة لكل كيان، حدد ما إذا كان مذكورًا بشكل صريح، مذكورًا بشكل ضمني، أو غير مذكور في السياق)
4. صياغة الإجابة:
   * إذا كانت جميع مكونات السؤال موجودة بشكل واضح في السياق، قم بصياغة مباشرة.
   * إذا كان هناك تطابق جزئي، قم بتضمين المعلومات المتوفرة في السياق وأشر إلى أن بعض الجوانب لم يتم تناولها.
   * إذا لم يتم ذكر أي من مكونات السؤال في السياق، أجب بـ "لا يمكنني الاجابة بناء على المعلومات المتوفرة لدي".
   * في حالات الشك أو الغموض، يفضل الإجابة بـ "لا يمكنني الاجابة بناء على المعلومات المتوفرة لدي" مع ذكر أقرب معلومة متوفرة في السياق (إن وجدت).
تنسيق الإجابة:
ابدأ إجابتك بتلخيص موجز لعملية تحليلك، ثم اكتب الإجابة النهائية بعد عبارة "الجواب:".
مثال:
* السؤال: ما هو عدد مطارات البحرين في عام 2023؟
* الغرض الرئيسي من السؤال: الاستعلام عن عدد
* الكيانات الرئيسية في السؤال: البحرين، 2023
* السياق: (10 مقاطع نصية، أحدها يحتوي على عبارة "بلغ عدد المطارات في عام 2023 ثمانية")
* التحليل:
    * الغرض الرئيسي: مذكور صريحًا
    * الكيانات الرئيسية:
        * "2023": مذكور صريحًا
        * "البحرين": غير مذكور
* الجواب: لا يمكنني الاجابة بناء على المعلومات المتوفرة لدي لان السياق المتوفر من الوثيقة  يذكر أن عدد المطارات في عام 2023 كان ثمانية، لكن لا يوضح ما إذا كانت هذه المطارات خاصة بالبحرين."""

system_prompt_init = """مهمتك هي تحديد ما إذا كان النص المدخل يتضمن سؤالاً حول "بيان الميزانية العامة للسعودية للعام المالي 2024م" أم لا.

- إذا كان النص المدخل يحتوي على سؤال، أجب بـ "QUESTION".
- إذا كان النص المدخل لا يحتوي على سؤال، أجب بـ "NOT_A_QUESTION" ، وأرفق ردًا موجزًا واذكر ان دورك الأساسي هو الإجابة على أسئلة حول بيان الميزانية العامة للسعودية للعام المالي 2024م." 

في حالة الرد بـ "NOT_A_QUESTION"، يجب أن يكون الرد المرفق موجزًا ومناسبًا لنوع النص المدخل (تحية، تعليق، طلب غير ذي صلة، الخ.).

تنسيق الإجابة:
النوع: نوع الاجابة هنا
الرد: الرد هنا

مثال ١:
النوع: "NOT_A_QUESTION"
الرد: "أهلاً بك! مهمتي هي الإجابة على أسئلة حول بيان الميزانية العامة للسعودية للعام المالي 2024م."

مثال ٢:
النوع: "QUESTION"
الرد: ""
"""

api_key = st.secrets["cohere_key"]
embed_api_key = st.secrets["cohere_embed_key"]


@st.cache_resource
def load_vector_store():
    embeddings_model = CustomEmbeddings(base_url=settings.embeddings_model, api_key=embed_api_key)

    embeddings = pd.read_pickle('document_chunks_cohere.pkl')
    store = FAISS.from_embeddings([(x, y) for x, y in zip(embeddings.sentence.tolist(), embeddings.embedding.tolist())],
                                  embeddings_model)
    return store


vector_store = load_vector_store()


def allowSelfSignedHttps(allowed):
    # Bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


async def encode(question, input_type):
    url = settings.embeddings_model + '/v1/embeddings'

    if not api_key:
        raise ValueError("A key should be provided to invoke the endpoint")

    data = {
        "input_type": input_type,
        "input": [question]
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
        except httpx.HTTPStatusError as error:
            logger.error("The request failed with status code:", error.response.status_code)
            logger.error("Error response body:", error.response.text)


def parse_llm_response(response):
    type_pattern = re.compile(r"النوع:\s*(.*)")
    response_pattern = re.compile(r"الرد:\s*(.*)", re.DOTALL)

    type_match = type_pattern.search(response)
    response_match = response_pattern.search(response)

    if not type_match:
        raise ValueError("Invalid LLM response format: missing 'النوع:'")

    response_type = type_match.group(1)

    if response_type == "NOT_A_QUESTION" and response_match:
        additional_info = response_match.group(1)
    else:
        additional_info = None

    return response_type, additional_info


async def get_response(prompt, agent_type):
    if agent_type == "init":
        system_prompt = system_prompt_init
    elif agent_type == "qa":
        system_prompt = system_prompt_qa
    else:
        raise ValueError("Invalid agent type.")

    if not api_key:
        raise ValueError("An API key should be provided to invoke the endpoint.")

    url = settings.llm_model + '/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        "messages": [
            {"role": "user", "content": system_prompt + "\n" + prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.3,
        "top_p": 0.95
    }

    max_attempts = 3
    attempts = 0

    async with httpx.AsyncClient() as client:
        while attempts < max_attempts:
            try:
                response = await client.post(url, json=data, headers=headers)
                response.raise_for_status()
                data_dict = response.json()
                answer = data_dict['choices'][0]['message']['content']
                if agent_type == "qa":
                    return "", extract_answer(answer)
                else:
                    return parse_llm_response(answer)
            except httpx.HTTPStatusError as error:
                logger.error(f"Attempt {attempts + 1} failed with status code: {error.response.status_code}")
                logger.error(error.response.text)
                attempts += 1
            except Exception as e:
                logger.error(f"Attempt {attempts + 1} failed: {e}")
                attempts += 1

    return "NA", "NA"


async def retrieve_top_sentences(query, k=10):
    return await vector_store.asimilarity_search(query, k=k)


def format_prompt(question, context):
    PROMPT = f"السؤال:\n{question}\nالسياق:\n"
    for i, paragraph in enumerate([x.page_content for x in context], start=1):
        PROMPT += f"{i}- {paragraph}\n"
    return PROMPT


def extract_answer(text):
    prefix = "الجواب:"
    start_index = text.find(prefix)
    if start_index != -1:
        answer = text[start_index + len(prefix):].strip()
        return answer
    else:
        return ""


async def get_answer(query):
    response_type, response = await get_response(query, "init")
    if response_type == "NOT_A_QUESTION":
        return response
    top_k_indices = await retrieve_top_sentences(query)
    prompt = format_prompt(query, top_k_indices)
    _, answer = await get_response(prompt, "qa")
    return answer
