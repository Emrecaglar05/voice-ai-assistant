import openai
from openai import OpenAI
import chromadb

# OpenAI istemci
client = OpenAI(api_key="BURAYA_API_ANAHTARINIZI_YAZIN")

# 1. Ses -> Metin (Whisper STT)
def ses_to_metin(ses_dosyasi):
    with open(ses_dosyasi, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

# 2. RAG (Vektör DB + ChatGPT)
def rag_cevap(soru):
    # Vektör DB (örnek: Chroma) -> burada yalnızca şablon gösteriliyor
    # Normalde belge ekleme ve arama yapılır
    chroma_client = chromadb.PersistentClient(path="vektor_db")
    koleksiyon = chroma_client.get_or_create_collection("baskan_bilgi")

    # benzer içerikleri getir
    benzer = koleksiyon.query(
        query_texts=[soru],
        n_results=3
    )

    icerikler = "\n".join(benzer.get("documents", [""]))

    # ChatGPT API çağrısı
    yanit = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Sen Dursun Mirza'nın klonusun. Resmi, anlaşılır ve net cevaplar ver."},
            {"role": "user", "content": f"Soru: {soru}\nİlgili içerikler: {icerikler}"}
        ]
    )

    return yanit.choices[0].message.content

# 3. Metin -> Ses (TTS)
def metin_to_ses(metin, cikti_dosya="cikti.wav"):
    ses = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=metin
    )
    with open(cikti_dosya, "wb") as f:
        f.write(ses.read())
    return cikti_dosya

# Ana pipeline
def baslat(ses_dosyasi):
    metin = ses_to_metin(ses_dosyasi)
    cevap = rag_cevap(metin)
    cikti_ses = metin_to_ses(cevap)
    return cikti_ses

# Örnek kullanım
if __name__ == "__main__":
    giris = "reference_speaker.wav"
    cikti = baslat(giris)
    print("Oluşturulan ses:", cikti)

