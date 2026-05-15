from TTS.api import TTS
import simpleaudio as sa
import tempfile
import os
from google import genai
import re
import torch
import speech_recognition as sr
import threading
import time
import noisereduce as nr
import numpy as np
import sys
import wave
from scipy.signal import butter, filtfilt
from collections import deque
import logging
from datetime import datetime
import random

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.stdout.reconfigure(encoding='utf-8')

# -------------------------
# TEMEL YAPILANDIRMA
# -------------------------
GOOGLE_API_KEY = "AIzaSyAOYE1dm7f30NNVkpyPy61XraUdwfqKzLw"
REFERENCE_SPEAKER = "C:/Desktop/Kayıt-_3_.wav"

# TTS modeli yükleme
print("🔥 Festival Asistan başlatılıyor...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)
print(f"🎯 Çalışma cihazı: {device}")

# Gemini AI client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Gelişmiş prompt
BASE_PROMPT = """
Sen festival ortamında seyircilerle konuşan bilgili ve eğlenceli bir REHBERsin. 
ASLA kendini "yapay zeka" olarak tanıtma - sen gerçek bir insansın!

Kurallar:
- Önceki konuşmaları hatırla ve ona göre yanıt ver
- Tarih sorusu geldiğinde: Bildiğin tarihi bilgiyi net ver, hikaye gibi anlat
- Atatürk sorusu: "Mustafa Kemal Atatürk! 1881'de Selanik'te doğdu, Çanakkale kahramanı!"
- Müzik sorusu: "Müzik ruhun gıdası! Bu festival atmosferinde hangi tür seviyorsun?"
- Savaş sorusu: Tarihi olayları kronolojik anlat 
- Festival havasında konuş: Enerjik ama bilgi dolu
- Maksimum 3 cümle, 180 karakter
- Bağlam kurarak devam eden konuşma yap

Örnekler:
- "Atatürk hakkında" → "Gazi Mustafa Kemal! Çanakkale'de muhteşem liderlik, sonra Cumhuriyeti kurdu!"
- "Müzik hakkında" → "Bu festival enerjisinde müzik konuşmak harika! Hangi tür favorin?"
- "Çanakkale" → "1915 Çanakkale Zaferi! 'Çanakkale geçilmez' sözünün gerçek anlamı!"
"""


# -------------------------
# Hafıza Sistemi (Basitleştirilmiş)
# -------------------------
class SimpleMemory:
    def __init__(self, max_history=15):
        self.history = deque(maxlen=max_history)
        self.topics = {}
        self.current_topic = None

    def add_exchange(self, user_input, ai_response):
        topic = self.extract_topic(user_input)
        exchange = {
            'user': user_input,
            'assistant': ai_response,
            'topic': topic,
            'time': datetime.now()
        }
        self.history.append(exchange)
        self.topics[topic] = self.topics.get(topic, 0) + 1
        self.current_topic = topic

    def extract_topic(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['atatürk', 'mustafa kemal']):
            return 'atatürk'
        elif any(word in text_lower for word in ['müzik', 'şarkı', 'parça']):
            return 'müzik'
        elif any(word in text_lower for word in ['çanakkale', 'savaş', 'muharebe']):
            return 'savaş'
        elif any(word in text_lower for word in ['tarih', 'geçmiş']):
            return 'tarih'
        else:
            return 'genel'

    def get_context(self):
        if not self.history:
            return ""
        recent = list(self.history)[-2:]  # Son 2 konuşma
        context = []
        for ex in recent:
            context.append(f"Önceki soru: '{ex['user'][:30]}...'")
        return "\n".join(context)


memory = SimpleMemory()


# -------------------------
# Cache Sistemi (Basitleştirilmiş)
# -------------------------
class SimpleCache:
    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size

    def _make_key(self, text):
        words = re.findall(r'\w+', text.lower())
        important_words = [w for w in words if len(w) > 3 and w not in ['hakkında', 'nedir', 'nasıl']]
        return ' '.join(sorted(set(important_words[:3])))

    def get(self, user_input):
        key = self._make_key(user_input)
        if key in self.cache:
            print(f"💾 Cache HIT: {key}")
            return self.cache[key]
        return None

    def set(self, user_input, response):
        if len(self.cache) >= self.max_size:
            # İlk eklenenileri sil
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        key = self._make_key(user_input)
        self.cache[key] = response
        print(f"💾 Cache SAVED: {key}")


cache = SimpleCache()


# -------------------------
# Ses İşleme (Basitleştirilmiş)
# -------------------------
class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000

    def process_audio(self, audio_data):
        # Basit gürültü azaltma
        try:
            reduced = nr.reduce_noise(y=audio_data, sr=self.sample_rate)
            # Normalize
            if np.max(np.abs(reduced)) > 0:
                return reduced / np.max(np.abs(reduced)) * 0.8
            return reduced
        except:
            return audio_data


audio_processor = AudioProcessor()

# -------------------------
# Global Değişkenler
# -------------------------
is_speaking = False
listening_active = True
current_play_obj = None

recognizer = sr.Recognizer()
recognizer.energy_threshold = 800
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 2.0
mic = sr.Microphone(sample_rate=16000, chunk_size=1024)


# -------------------------
# Ana Fonksiyonlar
# -------------------------
def get_ai_response(user_text):
    """Ana AI yanıt fonksiyonu - Basitleştirilmiş"""
    start_time = time.time()

    try:
        print(f"🧠 AI işlemi başladı: '{user_text}'")

        # Cache kontrolü
        cached = cache.get(user_text)
        if cached:
            return cached

        # Bağlam al
        context = memory.get_context()

        # Tam prompt oluştur
        full_prompt = f"{BASE_PROMPT}\n\n{context}\n\nKullanıcı diyor: '{user_text}'\n\nFestival rehberi yanıtı:"

        print("🚀 Gemini AI'ya gönderiliyor...")

        # Gemini'ye gönder
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=full_prompt
        )

        ai_text = response.text.strip()
        print(f"✅ Gemini yanıtı alındı: '{ai_text[:50]}...'")

        # Yanıt optimizasyonu
        sentences = re.split(r'[.!?]+', ai_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(sentences) > 3:
            ai_text = '. '.join(sentences[:3]) + '.'
        elif len(sentences) == 0:
            ai_text = "Bu konuda çok merak ediyorum! Biraz daha detay verir misin?"

        # Karakter limiti
        if len(ai_text) > 180:
            ai_text = ai_text[:177] + "..."

        # Cache ve hafızaya kaydet
        cache.set(user_text, ai_text)
        memory.add_exchange(user_text, ai_text)

        elapsed = time.time() - start_time
        print(f"⚡ AI yanıt süresi: {elapsed:.2f}s")

        return ai_text

    except Exception as e:
        logger.error(f"AI yanıt hatası: {e}")
        print(f"❌ AI Hatası: {e}")
        return "Bir sorun yaşadım ama hemen toparlanıyorum! Tekrar dener misin?"


def speak_text(text):
    global is_speaking, current_play_obj, listening_active

    if is_speaking and current_play_obj:
        try:
            current_play_obj.stop()
        except:
            pass

    is_speaking = True
    listening_active = False

    try:
        print(f"🎤 Konuşuyor: {text}")
        print("🔇 Dinleme DURDURULDU")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_wav = fp.name

        # TTS oluştur
        tts.tts_to_file(
            text=text,
            file_path=temp_wav,
            language="tr",
            speaker_wav=REFERENCE_SPEAKER,
            speed=1.1
        )

        # Oynat
        wave_obj = sa.WaveObject.from_wave_file(temp_wav)
        current_play_obj = wave_obj.play()
        current_play_obj.wait_done()

        # Temizle
        os.remove(temp_wav)

    except Exception as e:
        logger.error(f"TTS hatası: {e}")
        print(f"❌ TTS Hatası: {e}")
    finally:
        is_speaking = False
        current_play_obj = None

        print("⏳ Ses buffer temizleniyor...")
        time.sleep(1.0)

        # Mikrofon yenile
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print(f"🔊 Dinleme AKTİF - Eşik: {recognizer.energy_threshold:.1f}")
        except Exception as e:
            print(f"⚠️ Kalibrasyon uyarısı: {e}")

        listening_active = True


def listen_for_speech():
    try:
        if is_speaking or not listening_active:
            return None

        print("🎧 Dinliyorum...")

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)

        return audio

    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        logger.error(f"Dinleme hatası: {e}")
        return None


def process_audio(audio):
    try:
        # Audio verisini al
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32)

        # Ses işle
        clean_audio = audio_processor.process_audio(audio_data)
        processed_audio_int = (clean_audio * 32767).astype(np.int16)

        clean_audio_obj = sr.AudioData(
            processed_audio_int.tobytes(),
            audio.sample_rate,
            audio.sample_width
        )

        # Google STT
        results = recognizer.recognize_google(
            clean_audio_obj,
            language="tr-TR",
            show_all=True
        )

        if results and "alternative" in results:
            best_result = max(
                results["alternative"],
                key=lambda x: len(x["transcript"]) * x.get("confidence", 0.5)
            )

            user_input = best_result["transcript"]
            confidence = best_result.get("confidence", 0.5)

            print(f"👤 Kullanıcı: {user_input} (Güven: {confidence:.2f})")

            if confidence < 0.3:
                print("⚠️ Düşük güven, tekrar dinleniyor...")
                return None

            return user_input

        return None

    except sr.UnknownValueError:
        print("🤖 Ses anlaşılmadı")
        return None
    except Exception as e:
        logger.error(f"Ses işleme hatası: {e}")
        return None


def process_user_input(user_text):
    def worker():
        try:
            print(f"🔄 İşleniyor: '{user_text}'")
            ai_response = get_ai_response(user_text)
            print(f"✅ Yanıt hazır: '{ai_response}'")
            speak_text(ai_response)
        except Exception as e:
            logger.error(f"İşleme hatası: {e}")
            speak_text("Teknik bir sorun yaşadım, tekrar dener misin?")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


# -------------------------
# Başlangıç
# -------------------------
print("🎪 Festival Sesli Asistan (Düzeltilmiş Versiyon) başlatılıyor...")
print("🎯 Çıkmak için: 'quit', 'çıkış', 'kapat' diyebilirsiniz")

# Kalibrasyon
print("🔧 Mikrofon kalibrasyonu...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=3)

print(f"✅ Hazır! Gürültü eşiği: {recognizer.energy_threshold}")
print("🎉 FESTIVAL MODU AKTİF!")

# -------------------------
# Ana Döngü
# -------------------------
consecutive_errors = 0
total_interactions = 0

while True:
    try:
        if not listening_active or is_speaking:
            time.sleep(0.1)
            continue

        # Ses dinle
        audio = listen_for_speech()
        if audio is None:
            continue

        # Kontrol et
        if is_speaking or not listening_active:
            continue

        # Ses işle
        user_input = process_audio(audio)

        if user_input:
            consecutive_errors = 0
            total_interactions += 1

            # Çıkış kontrolü
            user_lower = user_input.lower()
            exit_words = ["quit", "çıkış", "kapat", "bitir", "son", "bye"]
            if any(word in user_lower for word in exit_words):
                speak_text("Festival sona erdi! Görüşürüz! 🎉")
                print(f"📊 Toplam {total_interactions} konuşma yapıldı!")
                if current_play_obj:
                    try:
                        current_play_obj.stop()
                    except:
                        pass
                break

            # Durum kontrolü
            elif any(word in user_lower for word in ["durum", "stats"]):
                speak_text(f"Bugün {total_interactions} konuşma yaptık! Harika gidiyor!")
                continue

            # Normal işlem
            print(f"🎯 İşleniyor: #{total_interactions}")
            process_user_input(user_input)

        else:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                speak_text("Ses gelmiyor! Mikrofona daha yakın konuş!")
                consecutive_errors = 0

    except KeyboardInterrupt:
        print(f"\n🛑 Durduruldu! Toplam {total_interactions} konuşma.")
        if current_play_obj:
            try:
                current_play_obj.stop()
            except:
                pass
        break
    except Exception as e:
        logger.error(f"Ana döngü hatası: {e}")
        print(f"❌ Sistem hatası: {e}")
        consecutive_errors += 1
        if consecutive_errors >= 5:
            print("🚨 Çok fazla hata! Sistem yeniden başlatılıyor...")
            time.sleep(2)
            consecutive_errors = 0

print("👋 Festival Sesli Asistan kapatıldı!")