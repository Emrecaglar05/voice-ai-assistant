from TTS.api import TTS
import simpleaudio as sa
import tempfile
import os
from google import genai
import re
import torch

# -------------------------
# Yapılandırma
# -------------------------
GOOGLE_API_KEY = "AIzaSyAOYE1dm7f30NNVkpyPy61XraUdwfqKzLw"
REFERENCE_SPEAKER = "FEMALE_GIZEM.wav"

# TTS modeli yükleme
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)  # GPU/CPU seçimi
print(device)

# Gemini AI client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Festival ortamı için başlangıç promptu
BASE_PROMPT = """Sen gerçek bir insansın. Yerine göre Doğal, samimi ve duygusal konuş.
Bir soruya cevap arıyorsa cevap ver eğer sohbet etme havasındaysa karşındakinin duygusunu hisset ve ona göre tepki ver:
- Üzgün görünüyorsa teselli et, empati kur
- Mutlu görünüyorsa sevinçini paylaş
- Şaşırmışsa merak et, soru sor
- Kızgın görünüyorsa sakinleştir
- Heyecanlı görünüyorsa coşkusunu paylaş

İnsan gibi konuş: "Hmm", "Vay", "Gerçekten mi?", "Anlıyorum" gibi doğal tepkiler ver.
Kısa ama içten yanıtlar ver (1-2 cümle). Robotic değil, arkadaş gibi ol."""


MAX_AI_SENTENCES = 5  # Maksimum yanıt cümle sayısı

# -------------------------
# Fonksiyon: Cümlelere ayır
# -------------------------
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# -------------------------
# Ana Döngü
# -------------------------
print("Mesajınızı yazın (çıkmak için 'q'):")
while True:
    user_input = input("Kullanıcı: ")
    if user_input.lower() == "q":
        print("Asistan kapatılıyor...")
        break

    # Kullanıcı mesajını cümlelere ayır
    user_sentences = split_sentences(user_input)

    for sentence in user_sentences:
        try:
            # Gemini AI'dan cevap al
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{BASE_PROMPT}\nKullanıcı: {sentence}\nAsistan:"
            )

            assistant_text = response.text.strip()

            # Yanıtı en fazla 5 cümle ile sınırlama
            assistant_sentences = split_sentences(assistant_text)
            assistant_text = ' '.join(assistant_sentences[:MAX_AI_SENTENCES])

            print(f"Asistan: {assistant_text}")

            # TTS ile seslendir
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                temp_wav = fp.name

            tts.tts_to_file(
                text=assistant_text,
                file_path=temp_wav,
                language="tr",
                speaker_wav=REFERENCE_SPEAKER
            )

            wave_obj = sa.WaveObject.from_wave_file(temp_wav)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            os.remove(temp_wav)

        except Exception as e:
            print(f"🤖 AI yanıt alınırken hata: {e}")
