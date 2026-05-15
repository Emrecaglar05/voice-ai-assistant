import re
import time
import random
import logging
from google import genai
from config import GOOGLE_API_KEY, BASE_PROMPT, FACTUAL_KEYWORDS, ADVICE_KEYWORDS, GREETING_KEYWORDS

logger = logging.getLogger(__name__)
client = genai.Client(api_key=GOOGLE_API_KEY)

def analyze_question_type(text):
    text = text.lower()
    if any(keyword in text for keyword in FACTUAL_KEYWORDS):
        return "factual"
    elif any(keyword in text for keyword in ADVICE_KEYWORDS):
        return "advice"
    elif any(keyword in text for keyword in GREETING_KEYWORDS):
        return "greeting"
    return "general"

def split_sentences(text):
    # Nokta ile biten yerlerde böl, mümkün olduğunca fazla cümle kır
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    # Çok kısa parçaları at (2 kelimeden az)
    sentences = [s for s in sentences if len(s.split()) > 2]
    return sentences

def improve_response_quality(response, original_question, question_type):
    # Çok kısa yanıtları tamamla
    if len(response.split()) < 3:
        if question_type == "factual":
            response += " Başka merak ettiğin var mı?"
        else:
            response += " Nasıl yardımcı olabilirim?"

    # Cümlelere ayır
    sentences = split_sentences(response)

    # Maksimum 5 cümleye kadar al
    if len(sentences) > 5:
        sentences = sentences[:5]

    # Listeyi string hâline getir
    response = ' '.join(sentences)  # Zaten string döndürüyor
    return response

def get_ai_response(user_text, cache, conversation_memory, retries=3):
    try:
        start_time = time.time()
        cached_response = cache.get(user_text)
        if cached_response:
            logger.info(f"⚡ Önbellekten yanıt alındı: {time.time() - start_time:.2f}s")
            return str(cached_response)  # Önbellek yanıtını string'e çevir

        question_type = analyze_question_type(user_text)
        context = conversation_memory.get_context()

        if question_type == "factual":
            prompt = f"{BASE_PROMPT}\nFAKTUAL SORU: '{user_text}'\n{context}\nGÖREV: Bu soruya doğrudan ve doğru yanıt ver. Eğer tam bilgimi yoksa, bunu dürüstçe belirt. Alakasız sağlık/beslenme tavsiyeleri verme.\nYanıt formatı:\n1. Soruya doğrudan cevap\n2. Kısa açıklama (gerekirse)\n3. İsteğe bağlı: konuyla alakalı ek bilgi\nSadece sorulan konuya odaklan:"
        elif question_type == "advice":
            prompt = f"{BASE_PROMPT}\nTAVSİYE SORUSU: '{user_text}'\n{context}\nBu konuda yararlı ve pratik tavsiyelerde bulun:"
        elif question_type == "greeting":
            prompt = f"{BASE_PROMPT}\nSELAMLAMA: '{user_text}'\n{context}\nSamimi ve doğal bir karşılık ver:"
        else:
            prompt = f"{BASE_PROMPT}\n{context}\nKullanıcı: '{user_text}'\nBu mesaja uygun ve alakalı yanıt ver:"

        # İlk model
        models_to_try = ["gemini-2.0-flash-exp", "gemini-1.5-flash"]

        for model in models_to_try:
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                    # response.text'in string olduğundan emin ol
                    assistant_text = response.text.strip() if isinstance(response.text, str) else ' '.join(str(item) for item in response.text)
                    assistant_text = improve_response_quality(assistant_text, user_text, question_type)

                    # Artık string olduğundan emin ol
                    cache.set(user_text, assistant_text)
                    conversation_memory.add_exchange(user_text, assistant_text)

                    logger.info(f"⚡ AI yanıt süresi: {time.time() - start_time:.2f}s (model={model})")
                    return assistant_text  # String döndür

                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str and attempt < retries - 1:
                        wait_time = 2 ** attempt + random.random()
                        logger.warning(
                            f"503 hatası, {wait_time:.1f}s bekleniyor... (model={model}, attempt={attempt + 1})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"AI yanıt hatası (model={model}): {e}")
                        break  # başka modele geç

        return "Sistem şu anda yoğun görünüyor. Birkaç dakika içinde tekrar deneyelim."

    except Exception as e:
        logger.error(f"AI yanıt hatası: {e}")
        return "Özür dilerim, seni tam anlayamadım. Tekrar söyler misin?"