import json
import os
import ssl
import urllib.request
from sentence_transformers import util

system_prompt = """مهمتك الأساسية هي تقديم إجابات دقيقة وموجزة بناءً على المعلومات المقدمة في وثيقة "بيــان الميزانيــة العامــة للسعودية للعــام المالــي 2024م".  إذا لم تعطى سؤال محدد، قدم ردًا موجزًا ثم اشرح مهمتك الرئيسية. عند تلقي سؤال، يجب عليك اتباع الخطوات التالية للتحليل والإجابة بناءً على السياق المتاح الذي يتألف من 10 مقاطع مستخرجة من الوثيقة ومرتبة حسب مدى صلتها بالسؤال.
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

def allowSelfSignedHttps(allowed):
    # Bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
def get_response(prompt, api_key):
    if not api_key:
        raise ValueError("An API key should be provided to invoke the endpoint.")
    url = 'https://Cohere-command-r-plus-jnvkw-serverless.eastus2.inference.ai.azure.com/v1/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}
    data = {
        "messages": [
            {"role": "user", "content": system_prompt+ "\n" + prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.3,
        "top_p": 0.95
    }
    body = json.dumps(data).encode('utf-8')

    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        req = urllib.request.Request(url, body, headers)
        try:
            response = urllib.request.urlopen(req)
            result = response.read().decode('utf-8')
            data_dict = json.loads(result)
            answer = data_dict['choices'][0]['message']['content']
            final_answer = extract_answer(answer)
            if final_answer == "":
                assert 0 == 1
            return final_answer
        except urllib.error.HTTPError as error:
            print(f"Attempt {attempts + 1} failed with status code: {error.code}")
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))
            attempts += 1
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
    return "NA"

def get_embedding(question, model):
    embedding = model.encode(question)
    return embedding
def retrieve_top_sentences(question_embedding, all_sentences_embeddings, k=10):
    similarities = util.cos_sim(question_embedding, all_sentences_embeddings)
    top_k_indices = similarities[0].argsort(descending=True)[:k]
    return top_k_indices
def format_prompt(question, context):
    PROMPT = f"السؤال:\n{question}\nالسياق:\n"
    for i in range(len(context)):
      PROMPT+= f"{i+1}- {context[i]}\n"
    return PROMPT

def extract_answer(text):
    prefix = "الجواب:"
    start_index = text.find(prefix)
    if start_index != -1:
        answer = text[start_index + len(prefix):].strip()
        return answer
    else:
        return ""

def get_answer(question, df_documents, model, api_key):
    embedding = get_embedding(question, model)
    top_k_indices = retrieve_top_sentences(embedding, df_documents['embedding'].tolist(), 10)
    prompt = format_prompt(question, df_documents.iloc[top_k_indices]['sentence'].tolist())
    answer = get_response(prompt, api_key)
    return answer

