from TTS.api import TTS
import simpleaudio as sa
import tempfile
import os
from google import genai
import re
import torch
import speech_recognition as sr
import pyaudio
import threading
import time
import noisereduce as nr
import numpy as np
import sys
import wave
from scipy.signal import butter, filtfilt
from collections import deque
import logging
import json
from datetime import datetime

# Loglama ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.stdout.reconfigure(encoding='utf-8')

# -------------------------
# Gelişmiş Yapılandırma
# -------------------------
GOOGLE_API_KEY = "AIzaSyAOYE1dm7f30NNVkpyPy61XraUdwfqKzLw"
REFERENCE_SPEAKER = "C:/Desktop/Kayıt-_3_.wav"

# TTS modeli yükleme
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)
print(f"🎯 Çalışma cihazı: {device}")

# Gemini AI client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Gelişmiş prompt
BASE_PROMPT = """
Sen konuşan bilgili ve eğlenceli bir REHBERsin. 
ASLA kendini "yapay zeka" olarak tanıtma - sen gerçek bir insansın!

Kurallar:
- aksium 3 cümle, 200 karakter
- Hiç kendini AI/robot olarak tanıtma!
- Bağlam kurarak devam eden konuşma yap

""Örnekler:
- "Atatürk hangi savaşlara katıldı?" → "Çanakkale'de muhteşem liderlik, sonra Kurtuluş Savaşı'nı yönetti!"
- "Şarkı söyle" → "En sevdiğim o güzel nakaratı hatırladın mı? La la la..."
"""""


# -------------------------
# Hafıza ve Hızlandırma Sistemleri
# -------------------------
class ConversationMemory:
    def __init__(self, max_history=10):
        self.history = deque(maxlen=max_history)
        self.context_keywords = {}
        self.current_topic = None

    def add_exchange(self, user_input, ai_response):
        """Konuşma geçmişini kaydet"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'assistant': ai_response,
            'topic': self.extract_topic(user_input)
        }
        self.history.append(exchange)
        self.update_context(user_input)

    def extract_topic(self, text):
        """Konuşmanın konusunu tespit et"""
        topics = {
            'atatürk': ['atatürk', 'mustafa kemal', 'cumhurbaşkanı'],
            'savaş': ['savaş', 'muharebe', 'çanakkale', 'kurtuluş'],
            'müzik': ['şarkı', 'müzik', 'parça', 'söyle'],
            'tarih': ['tarih', 'geçmiş', 'dönem', 'yıl'],
            'kişisel': ['nasılsın', 'kimsin', 'neler yapıyorsun']
        }

        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                self.current_topic = topic
                return topic

        return 'genel'

    def update_context(self, user_input):
        """Bağlam anahtar kelimelerini güncelle"""
        words = user_input.lower().split()
        for word in words:
            if len(word) > 3:  # Anlamlı kelimeleri say
                self.context_keywords[word] = self.context_keywords.get(word, 0) + 1

    def get_context(self):
        """Mevcut bağlamı döndür"""
        if not self.history:
            return ""

        # Son 3 konuşmayı özet olarak ver
        recent_context = []
        for exchange in list(self.history)[-3:]:
            recent_context.append(f"Kullanıcı: {exchange['user'][:50]}...")
            recent_context.append(f"Sen: {exchange['assistant'][:50]}...")

        context = "\n".join(recent_context)
        return f"\nÖnceki konuşma özeti:\n{context}\nMevcut konu: {self.current_topic or 'belirsiz'}\n"


class ResponseCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size

    def _generate_key(self, text):
        """Sorular için anahtar oluştur"""
        # Benzer soruları aynı keyde eşle
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        words = text_clean.split()
        # Önemli kelimeleri seç (stop words hariç)
        stop_words = {'bir', 'bu', 'şu', 'o', 'ben', 'sen', 've', 'ile', 'için', 'gibi', 'nasıl', 'ne', 'nedir'}
        key_words = [w for w in words if len(w) > 2 and w not in stop_words]
        return ' '.join(sorted(key_words[:3]))  # İlk 3 anahtar kelime

    def get(self, user_input):
        """Cache'den yanıt al"""
        key = self._generate_key(user_input)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            print(f"💾 Cache'den yanıt alındı: {key}")
            return self.cache[key]
        return None

    def set(self, user_input, response):
        """Cache'e yanıt kaydet"""
        if len(self.cache) >= self.max_size:
            # En az kullanılanı sil
            least_used = min(self.access_count.items(), key=lambda x: x[1])
            del self.cache[least_used[0]]
            del self.access_count[least_used[0]]

        key = self._generate_key(user_input)
        self.cache[key] = response
        self.access_count[key] = 1
        print(f"💾 Cache'e kaydedildi: {key}")


# Global hafıza ve cache objeleri
conversation_memory = ConversationMemory(max_history=15)
response_cache = ResponseCache(max_size=50)

# -------------------------
# Gelişmiş Global Değişkenler
# -------------------------
is_speaking = False
listening_active = True
current_play_obj = None
speech_buffer = deque(maxlen=50)
silence_threshold = 2.0
min_speech_duration = 1.0


class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 1024

    def apply_bandpass_filter(self, audio_data, lowcut=300, highcut=3400):
        """İnsan sesine odaklı bant geçiren filtre"""
        nyquist = self.sample_rate * 0.5
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, audio_data)

    def reduce_background_noise(self, audio_data, reduction_factor=0.8):
        """Gelişmiş gürültü azaltma"""
        try:
            # Stationary gürültü azaltma
            reduced = nr.reduce_noise(
                y=audio_data,
                sr=self.sample_rate,
                prop_decrease=reduction_factor,
                stationary=True
            )
            # Non-stationary gürültü azaltma (kalabalık sesi için)
            reduced = nr.reduce_noise(
                y=reduced,
                sr=self.sample_rate,
                prop_decrease=0.6,
                stationary=False
            )
            return reduced
        except Exception as e:
            logger.warning(f"Gürültü azaltma hatası: {e}")
            return audio_data

    def normalize_audio(self, audio_data):
        """Ses seviyesini normalize et"""
        if np.max(np.abs(audio_data)) > 0:
            return audio_data / np.max(np.abs(audio_data)) * 0.8
        return audio_data


audio_processor = AudioProcessor()


# -------------------------
# Gelişmiş Fonksiyonlar
# -------------------------
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?😊])\s+', text.strip())
    return [s for s in sentences if len(s.split()) > 2]


# Gelişmiş recognizer ayarları
recognizer = sr.Recognizer()
recognizer.energy_threshold = 800
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 2.5
recognizer.phrase_threshold = 0.3
recognizer.non_speaking_duration = 1.2

# Mikrofon ayarları
mic = sr.Microphone(sample_rate=16000, chunk_size=1024)


def get_ai_response(user_text):
    try:
        start_time = time.time()

        # Önce cache'e bak
        cached_response = response_cache.get(user_text)
        if cached_response:
            return cached_response

        # Bağlam bilgisini al
        context = conversation_memory.get_context()

        # AI'dan yanıt al
        prompt = f"{BASE_PROMPT}\n{context}\nFestival ortamında seyirci diyor ki: '{user_text}'\n\nÖnceki konuşmayı hatırlayarak bağlamsal yanıt ver:"

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )

        assistant_text = response.text.strip()

        # Yanıt optimizasyonu
        assistant_sentences = split_sentences(assistant_text)
        if len(assistant_sentences) > 3:
            assistant_text = ' '.join(assistant_sentences[:3])
        else:
            assistant_text = ' '.join(assistant_sentences)

        if len(assistant_text.split()) < 3:
            assistant_text += " Bu konuda ne düşünüyorsun?"

        if len(assistant_text) > 200:
            assistant_text = assistant_text[:197] + "..."

        # Cache'e kaydet
        response_cache.set(user_text, assistant_text)

        # Hafızaya kaydet
        conversation_memory.add_exchange(user_text, assistant_text)

        elapsed_time = time.time() - start_time
        print(f"⚡ AI yanıt süresi: {elapsed_time:.2f}s")

        return assistant_text

    except Exception as e:
        logger.error(f"AI yanıt hatası: {e}")
        return "Özür dilerim, seni tam anlayamadım. Tekrar söyler misin?"


def speak_text(text):
    global is_speaking, current_play_obj, listening_active

    # Önceki konuşmayı durdur
    if is_speaking:
        if current_play_obj and current_play_obj.is_playing():
            current_play_obj.stop()

    is_speaking = True
    listening_active = False

    try:
        print(f"🎤 Asistan: {text}")
        print("🔇 Dinleme DURDURULDU - Asistan konuşuyor...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_wav = fp.name

        # TTS oluşturma
        tts.tts_to_file(
            text=text,
            file_path=temp_wav,
            language="tr",
            speaker_wav=REFERENCE_SPEAKER,
            speed=1.1
        )

        wave_obj = sa.WaveObject.from_wave_file(temp_wav)
        current_play_obj = wave_obj.play()
        current_play_obj.wait_done()

        os.remove(temp_wav)

    except Exception as e:
        logger.error(f"TTS hatası: {e}")
    finally:
        is_speaking = False
        current_play_obj = None

        # Asistan sesinin yankılanmasını önlemek için buffer
        print("⏳ Ses temizleniyor...")
        time.sleep(1.5)

        # Mikrofon kalibrasyonunu yenile
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print(f"🔊 Dinleme YENİDEN AKTİF - Eşik: {recognizer.energy_threshold:.1f}")
        except Exception as e:
            logger.warning(f"Kalibrasyon hatası: {e}")

        listening_active = True


def advanced_listen_for_speech():
    """Gelişmiş ses dinleme fonksiyonu"""
    try:
        if is_speaking or not listening_active:
            return None

        print("🎧 Dinliyorum... (Festival modunda)")

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            audio = recognizer.listen(
                source,
                timeout=6,
                phrase_time_limit=20
            )

        return audio

    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        logger.error(f"Dinleme hatası: {e}")
        return None


def process_audio(audio):
    """Ses işleme ve tanıma"""
    try:
        # Ses verisini numpy array'e çevir
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32)

        # Ses işleme pipeline
        filtered_audio = audio_processor.apply_bandpass_filter(audio_data)
        clean_audio = audio_processor.reduce_background_noise(filtered_audio)
        normalized_audio = audio_processor.normalize_audio(clean_audio)
        processed_audio_int = (normalized_audio * 32767).astype(np.int16)

        clean_audio_data = sr.AudioData(
            processed_audio_int.tobytes(),
            audio.sample_rate,
            audio.sample_width
        )

        # Google STT ile tanıma
        results = recognizer.recognize_google(
            clean_audio_data,
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
                speak_text("Seni tam anlayamadım, biraz daha yüksek sesle söyler misin?")
                return None

            return user_input
        else:
            return None

    except sr.UnknownValueError:
        print("🤖 Ses anlaşılmadı")
        return None
    except Exception as e:
        logger.error(f"Ses işleme hatası: {e}")
        return None


def process_user_input(user_text):
    """Kullanıcı girdisini işle"""

    def worker():
        try:
            ai_response = get_ai_response(user_text)
            speak_text(ai_response)
        except Exception as e:
            logger.error(f"Yanıt işleme hatası: {e}")
            speak_text("Bir sorun yaşadım, tekrar dener misin?")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def show_memory_stats():
    """Hafıza istatistiklerini göster"""
    print(f"\n📊 Hafıza Durumu:")
    print(f"   Konuşma geçmişi: {len(conversation_memory.history)} exchange")
    print(f"   Cache boyutu: {len(response_cache.cache)} yanıt")
    print(f"   Mevcut konu: {conversation_memory.current_topic or 'belirsiz'}")
    print(f"   Bağlam kelimeleri: {len(conversation_memory.context_keywords)} adet\n")


# -------------------------
# Başlangıç ve Kalibrasyon
# -------------------------
print("🎪 Hafızalı Festival Sesli Asistan başlatılıyor...")
print("🧠 Konuşma hafızası ve yanıt cache'i aktif!")
print("🎯 Çıkmak için 'quit', 'çıkış', 'kapat' veya 'bitir' diyebilirsiniz.")

print("🔧 Ortam sesi kalibrasyonu yapılıyor...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=8)

print(f"✅ Kalibrasyon tamamlandı. Gürültü eşiği: {recognizer.energy_threshold}")
print("🎉 Festival modu aktif! Seyirciler sorularını sorabilir.")

# -------------------------
# Ana Döngü
# -------------------------
consecutive_errors = 0
max_consecutive_errors = 5

while True:
    try:
        # Asistan konuşurken kesinlikle dinleme yapma
        if not listening_active or is_speaking:
            time.sleep(0.2)
            continue

        # Ses dinleme
        audio = advanced_listen_for_speech()

        if audio is None:
            time.sleep(0.1)
            continue

        # Tekrar kontrol et
        if is_speaking or not listening_active:
            continue

        # Ses işleme ve tanıma
        user_input = process_audio(audio)

        if user_input:
            consecutive_errors = 0

            # Çıkış komutları kontrolü
            exit_words = ["quit", "çıkış", "kapat", "bitir", "son", "bye"]
            if any(word in user_input.lower() for word in exit_words):
                speak_text("Festival sona erdi! Görüşürüz!")
                show_memory_stats()  # Son durumu göster
                print("🎪 Asistan kapatılıyor...")
                if current_play_obj and current_play_obj.is_playing():
                    current_play_obj.stop()
                break

            # Kullanıcı girdisini işle
            process_user_input(user_input)
        else:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                speak_text("Ses kalitesi düşük. Mikrofona daha yakın konuş!")
                consecutive_errors = 0

        # Her 10 konuşmada bir hafıza durumu göster
        if len(conversation_memory.history) > 0 and len(conversation_memory.history) % 10 == 0:
            show_memory_stats()

    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından durduruldu")
        show_memory_stats()
        if current_play_obj and current_play_obj.is_playing():
            current_play_obj.stop()
        break
    except Exception as e:
        logger.error(f"Ana döngü hatası: {e}")
        consecutive_errors += 1
        if consecutive_errors >= max_consecutive_errors:
            print("🚨 Çok fazla hata! Sistem yeniden başlatılıyor...")
            consecutive_errors = 0
            time.sleep(2)

print("👋 Hafızalı Festival Sesli Asistan kapatıldı!")