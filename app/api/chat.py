# from fastapi import APIRouter
# from pydantic import BaseModel
# from app.core.rag import retrieve_context
# from app.core.llm import generate_answer
# from app.core.config import DISTANCE_THRESHOLD
# import traceback
#
# router = APIRouter()
#
# class ChatRequest(BaseModel):
#     message: str
#
# @router.post("/chat")
# def chat(request: ChatRequest):
#     try:
#         query = request.message
#         print("👉 Received query:", query)
#
#         best_distance, context = retrieve_context(query)
#         print("👉 Distance:", best_distance)
#
#         if best_distance < DISTANCE_THRESHOLD:
#             prompt = f"""
# You are an agriculture expert.
# Answer ONLY using the context below.
# Do not give pesticide dosages.
#
# Context:
# {context}
#
# Question:
# {query}
# """
#             source = "RAG"
#         else:
#             prompt = f"""
# You are an agriculture assistant.
# Answer using general agriculture knowledge.
# Do not give pesticide dosages.
#
# Question:
# {query}
# """
#             source = "LLM"
#
#         answer = generate_answer(prompt)
#         print("👉 LLM answered")
#
#         return {
#             "reply": answer,
#             "source": source
#         }
#
#     except Exception as e:
#         print("❌ ERROR OCCURRED")
#         traceback.print_exc()
#         return {
#             "error": str(e)
#         }
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.rag import retrieve_context
from app.core.llm import generate_answer
from app.core.config import DISTANCE_THRESHOLD

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    language: str = "en"   # "en" for English, "hi" for Hindi


@router.post("/chat")
def chat(request: ChatRequest):
    query = request.message
    lang = request.language.lower()

    best_distance, context = retrieve_context(query)

    # ---------- RAG MODE ----------
    if best_distance < DISTANCE_THRESHOLD:
        if lang == "hi":
            prompt = f"""
आप एक कृषि विशेषज्ञ हैं।
केवल नीचे दिए गए संदर्भ का उपयोग करके उत्तर दें।
उत्तर हिंदी में दें।
कीटनाशक की मात्रा या खुराक न बताएं।
यदि जानकारी उपलब्ध न हो तो स्पष्ट रूप से बताएं।

संदर्भ:
{context}

प्रश्न:
{query}
"""
            source = "RAG"
        else:
            prompt = f"""
You are an agriculture expert.
Answer ONLY using the context below.
Answer in English.
Do not give pesticide dosages.
If information is not available, say so clearly.

Context:
{context}

Question:
{query}
"""
            source = "RAG"

    # ---------- LLM FALLBACK ----------
    else:
        if lang == "hi":
            prompt = f"""
आप एक कृषि सहायक हैं।
सामान्य कृषि ज्ञान का उपयोग करके उत्तर दें।
उत्तर हिंदी में दें।
कीटनाशक की मात्रा न बताएं।
यदि निश्चित न हों, तो स्पष्ट रूप से बताएं।

प्रश्न:
{query}
"""
            source = "LLM"
        else:
            prompt = f"""
You are an agriculture assistant.
Answer using general agriculture knowledge.
Answer in English.
Do not give pesticide dosages.
If unsure, say so.

Question:
{query}
"""
            source = "LLM"

    try:
        answer = generate_answer(prompt)
    except Exception:
        return {
            "reply": "The system is currently busy. Please try again later.",
            "source": "ERROR"
        }

    return {
        "reply": answer,
        "source": source
    }
