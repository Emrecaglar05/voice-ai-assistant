# -*- coding: utf-8 -*-
"""
Gelişmiş Sesli Asistan Sistemi

Bu sistem, aşağıdaki ana bileşenleri kullanarak bir sesli asistan işlevi görür:
1.  **Konuşmayı Metne Çevirme (STT):** Whisper modeli ile sesi metne dönüştürür.
2.  **Yapay Zeka Yanıt Üretimi:** Google Gemini AI ile metin girdisine anlamlı yanıtlar oluşturur.
3.  **Metni Sese Çevirme (TTS):** Coqui TTS ile üretilen metni seslendirir.
4.  **Verimlilik Sistemleri:**
    - Asenkron işlem hattı (pipeline) ile eş zamanlı görev yönetimi.
    - Akıllı önbellek (semantic cache) ile sık sorulan sorulara anında yanıt.
"""

# Gerekli Kütüphaneler
import os
import sys
import time
import logging
import asyncio
import tempfile
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
import re

# Veri İşleme ve Ses
import numpy as np
import noisereduce as nr
from scipy.signal import butter, filtfilt

# AI ve Ses Tanıma
import whisper
import speech_recognition as sr
from google import genai
from TTS.api import TTS

# Yardımcı Kütüphaneler
from dotenv import load_dotenv
from difflib import SequenceMatcher

from spacy.lang import sa

# -----------------------------------------------------------------------------
# 1. YAPILANDIRMA VE SABİTLER
# -----------------------------------------------------------------------------

# Çevre değişkenlerini yükle (.env dosyasından)
load_dotenv()

# Logger ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Türkçe karakter sorunları için stdout yapılandırması
sys.stdout.reconfigure(encoding='utf-8')


class Config:
    """Uygulama genelindeki tüm yapılandırma ayarlarını ve sabitleri barındırır."""
    # API Anahtarları
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    # Dosya Yolları
    REFERENCE_SPEAKER_PATH = os.path.join(os.getcwd(), "audio", "reference_speaker.wav")

    # Modeller
    TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    WHISPER_MODEL = "base"  # "tiny", "base", "small", "medium", "large"
    GEMINI_MODEL = "gemini-1.5-flash"

    # Ses İşleme Ayarları
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    LOWCUT_FILTER = 300
    HIGHCUT_FILTER = 3400

    # Ses Tanıma Eşikleri
    ENERGY_THRESHOLD = 800
    PAUSE_THRESHOLD = 2.0
    PHRASE_THRESHOLD = 0.3
    NON_SPEAKING_DURATION = 1.0
    CONFIDENCE_THRESHOLD = 0.3

    # Önbellek Ayarları
    CACHE_MAX_SIZE = 150
    CACHE_SIMILARITY_THRESHOLD = 0.70

    # Konuşma Geçmişi
    CONVERSATION_MAX_HISTORY = 10

    # Pipeline
    MAX_WORKERS = 4

    # Gemini AI için Temel Prompt
    BASE_PROMPT = """
    Sen bilgili ve yardımsever bir asistansın. Kullanıcının sorduğu soruya doğrudan, kısa ve net bir yanıt ver.
    Cevapların en fazla 2-3 cümle ve yaklaşık 180 karakter olmalıdır.
    - Felsefi veya aşırı detaylı cevaplardan kaçın.
    - Bilmediğin konularda "Bu konuda bilgim yok." de.
    - Konuşma dilinde, doğal ve samimi bir üslup kullan.
    """


# -----------------------------------------------------------------------------
# 2. SİSTEM BAŞLATMA VE KONTROL
# -----------------------------------------------------------------------------

def initialize_ai_services():
    """Gerekli AI modellerini ve istemcilerini yükler."""
    # Cihaz belirleme (GPU veya CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Kullanılacak cihaz: {device}")

    # TTS Modeli
    try:
        tts_client = TTS(Config.TTS_MODEL)
        tts_client.to(device)
        logger.info("TTS modeli başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"TTS modeli yüklenirken hata oluştu: {e}")
        tts_client = None

    # Whisper Modeli
    try:
        whisper_model = whisper.load_model(Config.WHISPER_MODEL)
        logger.info("Whisper (STT) modeli başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"Whisper modeli yüklenirken hata oluştu: {e}")
        whisper_model = None

    # Gemini AI İstemcisi
    gemini_client = None
    if Config.GOOGLE_API_KEY:
        try:
            gemini_client = genai.Client(api_key=Config.GOOGLE_API_KEY)
            logger.info("Google AI Studio (Gemini) istemcisi başarıyla başlatıldı.")
        except Exception as e:
            logger.error(f"Gemini istemcisi başlatılamadı: {e}")
    else:
        logger.warning("GOOGLE_API_KEY bulunamadı. Gemini AI devre dışı.")

    if not os.path.exists(Config.REFERENCE_SPEAKER_PATH):
        logger.error(f"Referans ses dosyası bulunamadı: {Config.REFERENCE_SPEAKER_PATH}")

    return tts_client, whisper_model, gemini_client


# Global AI istemcilerini başlatma
tts, whisper_model, client = initialize_ai_services()
is_speaking = threading.Event()
listening_active = threading.Event()
listening_active.set()
current_play_obj = None


# -----------------------------------------------------------------------------
# 3. ÇEKİRDEK SINIFLAR (Önbellek, Hafıza, Ses İşleme)
# -----------------------------------------------------------------------------

class SmartSemanticCache:
    """Benzer anlama gelen sorular için AI yanıtlarını önbelleğe alır."""

    def __init__(self, max_size, similarity_threshold):
        self.cache = {}
        self.semantic_cache = {}
        self.access_stats = {}
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.stats = {'hits': 0, 'misses': 0, 'semantic_hits': 0}

    def _get_cache_key(self, text):
        """Metinden anlamlı bir önbellek anahtarı oluşturur."""
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        return ' '.join(sorted(set(clean_text.split()))[:5])

    def _calculate_similarity(self, text1, text2):
        """İki metin arasındaki anlamsal benzerliği hesaplar."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def get(self, user_input):
        """Verilen girdi için önbellekte bir yanıt arar."""
        cache_key = self._get_cache_key(user_input)
        if cache_key in self.cache:
            self.stats['hits'] += 1
            self.access_stats[cache_key]['count'] += 1
            self.access_stats[cache_key]['last_access'] = time.time()
            logger.info(f"Önbellek (tam eşleşme): '{cache_key}' bulundu.")
            return self.cache[cache_key]

        # Anlamsal olarak en yakın sonucu bul
        best_match, best_score = None, 0.0
        for cached_input, response in self.semantic_cache.items():
            similarity = self._calculate_similarity(user_input, cached_input)
            if similarity > best_score:
                best_score, best_match = similarity, response

        if best_match and best_score >= self.similarity_threshold:
            self.stats['semantic_hits'] += 1
            logger.info(f"Önbellek (semantik eşleşme): Skor {best_score:.2f} ile bulundu.")
            return best_match

        self.stats['misses'] += 1
        return None

    def set(self, user_input, response):
        """Yeni bir AI yanıtını önbelleğe ekler."""
        if len(self.cache) >= self.max_size:
            self._evict()

        cache_key = self._get_cache_key(user_input)
        self.cache[cache_key] = response
        self.semantic_cache[user_input] = response
        self.access_stats[cache_key] = {'count': 1, 'last_access': time.time()}
        logger.info(f"Önbelleğe eklendi: '{cache_key}'")

    def _evict(self):
        """Önbellek dolduğunda en az kullanılan veriyi siler."""
        if not self.cache:
            return
        # En eski ve en az erişilen kaydı sil
        lru_key = min(self.access_stats.items(), key=lambda x: x[1]['last_access'])[0]
        del self.cache[lru_key]
        # semantic_cache'den de ilgili girdiyi bulup silmek gerekir, bu versiyonda basit tutulmuştur.
        del self.access_stats[lru_key]
        logger.info(f"Önbellekten silindi (eviction): '{lru_key}'")

    def preload_common_responses(self):
        """Sık kullanılan basit komutları önbelleğe yükler."""
        common_qa = {
            "merhaba": "Merhaba, size nasıl yardımcı olabilirim?",
            "nasılsın": "İyiyim, teşekkürler. Umarım siz de iyisinizdir.",
            "teşekkür ederim": "Rica ederim, başka bir konuda yardımcı olabilirim.",
            "görüşürüz": "Görüşmek üzere, iyi günler!"
        }
        for q, a in common_qa.items():
            self.set(q, a)
        logger.info(f"{len(common_qa)} yaygın yanıt önbelleğe yüklendi.")


class ConversationMemory:
    """Konuşma geçmişini ve bağlamı yönetir."""

    def __init__(self, max_history):
        self.history = deque(maxlen=max_history)

    def add_exchange(self, user_input, ai_response):
        """Yeni bir kullanıcı-asistan etkileşimini hafızaya ekler."""
        self.history.append({'user': user_input, 'assistant': ai_response})

    def get_context(self):
        """Model için geçmiş konuşmalardan bir bağlam metni oluşturur."""
        if not self.history:
            return ""
        context = "\nÖnceki Konuşma Bağlamı:\n"
        # Son iki etkileşimi al
        for exchange in list(self.history)[-2:]:
            context += f"- Kullanıcı: {exchange['user']}\n- Asistan: {exchange['assistant']}\n"
        return context


class AudioProcessor:
    """Gelen ham ses verisini temizler ve işler."""

    def __init__(self, sample_rate, lowcut, highcut):
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut

    def process(self, raw_data):
        """Sese gürültü azaltma, filtreleme ve normalizasyon uygular."""
        audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Gürültü Azaltma
        try:
            reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=self.sample_rate)
        except Exception:
            reduced_noise_audio = audio_data  # Hata olursa orijinal veriyi kullan

        # Bant Geçiren Filtre (İnsan sesi frekans aralığı)
        nyquist = 0.5 * self.sample_rate
        b, a = butter(4, [self.lowcut / nyquist, self.highcut / nyquist], btype='band')
        filtered_audio = filtfilt(b, a, reduced_noise_audio)

        # Normalizasyon
        max_val = np.max(np.abs(filtered_audio))
        if max_val > 0:
            normalized_audio = filtered_audio / max_val * 0.9
        else:
            normalized_audio = filtered_audio

        return (normalized_audio * 32767).astype(np.int16)


# -----------------------------------------------------------------------------
# 4. TEMEL FONKSİYONLAR (STT, AI, TTS, Dinleme)
# -----------------------------------------------------------------------------

def process_speech_to_text(audio_data):
    """Ses verisini metne çevirir."""
    if not whisper_model or not audio_data:
        return None

    try:
        processed_audio_bytes = audio_processor.process(audio_data.get_raw_data()).tobytes()

        # Whisper'ın işleyebilmesi için geçici WAV dosyası oluştur
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
            tmp_file.write(audio_data.get_wav_data())  # İşlenmemiş orijinal veriyi kullanmak daha iyi sonuç verebilir
            result = whisper_model.transcribe(tmp_file.name, language="tr", fp16=torch.cuda.is_available())

        text = result["text"].strip()
        logger.info(f"STT Sonucu: '{text}'")

        # Güvenilirlik düşükse veya metin anlamsızsa atla
        if not text or len(text) < 3:
            return None

        return text
    except Exception as e:
        logger.error(f"Ses işleme (STT) sırasında hata: {e}")
        return None


def generate_ai_response(user_text):
    """Kullanıcı metnine göre AI yanıtı üretir."""
    if not client:
        return "Üzgünüm, yapay zeka servisine şu anda ulaşılamıyor."

    # Önce önbelleği kontrol et
    cached_response = smart_cache.get(user_text)
    if cached_response:
        return cached_response

    try:
        # Prompt'u bağlam ile zenginleştir
        context = conversation_memory.get_context()
        prompt = f"{Config.BASE_PROMPT}{context}\nKullanıcı Sorusu: '{user_text}'"

        response = client.generate_content(model=Config.GEMINI_MODEL, contents=prompt)
        ai_text = response.text.strip()

        # Yanıtı hafızaya ve önbelleğe ekle
        smart_cache.set(user_text, ai_text)
        conversation_memory.add_exchange(user_text, ai_text)

        logger.info(f"AI Yanıtı: '{ai_text}'")
        return ai_text
    except Exception as e:
        logger.error(f"AI yanıtı oluşturulurken hata: {e}")
        return "Üzgünüm, bir sorun oluştu. Lütfen tekrar dener misiniz?"


def play_audio_response(text):
    """Verilen metni seslendirir."""
    global current_play_obj
    if not tts or not text:
        return

    is_speaking.set()
    listening_active.clear()

    try:
        # Eğer önceki ses çalmaya devam ediyorsa durdur
        if current_play_obj and current_play_obj.is_playing():
            current_play_obj.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_wav_path = fp.name

        tts.tts_to_file(
            text=text,
            file_path=temp_wav_path,
            language="tr",
            speaker_wav=Config.REFERENCE_SPEAKER_PATH
        )

        wave_obj = sa.WaveObject.from_wave_file(temp_wav_path)
        current_play_obj = wave_obj.play()
        current_play_obj.wait_done()

    except Exception as e:
        logger.error(f"Metin seslendirilirken (TTS) hata: {e}")
    finally:
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        is_speaking.clear()
        listening_active.set()
        # Konuşma bittikten sonra ortam gürültüsünü tekrar ayarla
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)


def listen_for_speech():
    """Mikrofondan ses dinler ve ses verisi döndürür."""
    if is_speaking.is_set() or not listening_active.is_set():
        return None

    try:
        with mic as source:
            logger.info("Dinliyorum...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            return audio
    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        logger.error(f"Dinleme sırasında hata: {e}")
        return None


# -----------------------------------------------------------------------------
# 5. ASENKRON İŞLEM HATTI (PIPELINE)
# -----------------------------------------------------------------------------

class AsyncPipelineManager:
    """STT, AI ve TTS işlemlerini paralel olarak yönetir."""

    def __init__(self, max_workers):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.audio_queue = asyncio.Queue(maxsize=5)
        self.ai_queue = asyncio.Queue(maxsize=5)
        self.tts_queue = asyncio.Queue(maxsize=5)
        self.is_running = True

    async def start(self):
        """Tüm işçi (worker) görevlerini başlatır."""
        logger.info("Asenkron işlem hattı başlatılıyor...")
        tasks = [
            asyncio.create_task(self.audio_worker()),
            asyncio.create_task(self.ai_worker()),
            asyncio.create_task(self.tts_worker())
        ]
        await asyncio.gather(*tasks)

    def stop(self):
        """İşlem hattını güvenli bir şekilde durdurur."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Asenkron işlem hattı durduruldu.")

    async def add_audio_to_queue(self, audio_data):
        """İşlenmek üzere ses verisini kuyruğa ekler."""
        if not self.audio_queue.full():
            await self.audio_queue.put(audio_data)
        else:
            logger.warning("Ses işleme kuyruğu dolu. Girdi atlanıyor.")

    async def audio_worker(self):
        """Ses verisini kuyruktan alıp metne çevirir."""
        loop = asyncio.get_event_loop()
        while self.is_running:
            try:
                audio = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                text = await loop.run_in_executor(self.executor, process_speech_to_text, audio)
                if text:
                    await self.ai_queue.put(text)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audio Worker hatası: {e}")

    async def ai_worker(self):
        """Metni kuyruktan alıp AI yanıtı oluşturur."""
        loop = asyncio.get_event_loop()
        while self.is_running:
            try:
                text = await asyncio.wait_for(self.ai_queue.get(), timeout=1.0)
                response = await loop.run_in_executor(self.executor, generate_ai_response, text)
                if response:
                    await self.tts_queue.put(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"AI Worker hatası: {e}")

    async def tts_worker(self):
        """Yanıt metnini kuyruktan alıp seslendirir."""
        loop = asyncio.get_event_loop()
        while self.is_running:
            try:
                text = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)
                await loop.run_in_executor(self.executor, play_audio_response, text)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"TTS Worker hatası: {e}")


# -----------------------------------------------------------------------------
# 6. ANA UYGULAMA DÖNGÜLERİ
# -----------------------------------------------------------------------------

async def run_asynchronous_loop(pipeline):
    """Asenkron ana döngü. Ses dinler ve pipeline'a gönderir."""
    pipeline_task = asyncio.create_task(pipeline.start())
    loop = asyncio.get_event_loop()
    try:
        while True:
            audio = await loop.run_in_executor(pipeline.executor, listen_for_speech)
            if audio:
                await pipeline.add_audio_to_queue(audio)
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Durdurma sinyali alındı.")
    finally:
        pipeline.stop()
        pipeline_task.cancel()


def run_synchronous_loop():
    """Basit, sıralı işlem yapan senkron ana döngü."""
    while True:
        try:
            audio = listen_for_speech()
            if audio:
                user_input = process_speech_to_text(audio)
                if user_input:
                    # Çıkış komutları
                    if any(word in user_input.lower() for word in ["çıkış", "kapat", "bitir"]):
                        play_audio_response("Görüşmek üzere, iyi günler!")
                        break

                    ai_response = generate_ai_response(user_input)
                    play_audio_response(ai_response)
            time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Program durduruldu.")
            break
        except Exception as e:
            logger.error(f"Senkron döngüde bir hata oluştu: {e}")
            time.sleep(1)


# -----------------------------------------------------------------------------
# 7. SİSTEM BAŞLATMA VE GİRİŞ NOKTASI
# -----------------------------------------------------------------------------

def system_setup():
    """Gerekli nesneleri ve ayarları yapar."""
    global recognizer, mic, smart_cache, conversation_memory, audio_processor

    # Ses Tanıma
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = Config.ENERGY_THRESHOLD
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = Config.PAUSE_THRESHOLD
    recognizer.phrase_threshold = Config.PHRASE_THRESHOLD
    recognizer.non_speaking_duration = Config.NON_SPEAKING_DURATION
    mic = sr.Microphone(sample_rate=Config.SAMPLE_RATE, chunk_size=Config.CHUNK_SIZE)

    with mic as source:
        logger.info("Ortam gürültüsü kalibrasyonu yapılıyor, lütfen sessiz olun...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        logger.info(f"Kalibrasyon tamamlandı. Gürültü eşiği: {recognizer.energy_threshold:.2f}")

    # Yardımcı Sınıflar
    smart_cache = SmartSemanticCache(
        max_size=Config.CACHE_MAX_SIZE,
        similarity_threshold=Config.CACHE_SIMILARITY_THRESHOLD
    )
    smart_cache.preload_common_responses()

    conversation_memory = ConversationMemory(max_history=Config.CONVERSATION_MAX_HISTORY)
    audio_processor = AudioProcessor(
        sample_rate=Config.SAMPLE_RATE,
        lowcut=Config.LOWCUT_FILTER,
        highcut=Config.HIGHCUT_FILTER
    )

    logger.info("Sesli Asistan Sistemi başlatılmaya hazır.")


if __name__ == "__main__":
    if not all([tts, whisper_model, client]):
        logger.error("Gerekli AI modellerinden biri veya birkaçı yüklenemedi. Program sonlandırılıyor.")
        sys.exit(1)

    system_setup()

    # Başlangıç Mesajı
    play_audio_response("Asistanınız hizmete hazır.")

    while True:
        mode = input("Çalışma modunu seçin [1: Asenkron (Önerilen), 2: Senkron]: ").strip()
        if mode == '1':
            pipeline = AsyncPipelineManager(max_workers=Config.MAX_WORKERS)
            try:
                asyncio.run(run_asynchronous_loop(pipeline))
            except KeyboardInterrupt:
                logger.info("Asenkron döngü sonlandırıldı.")
            break
        elif mode == '2':
            logger.info("Klasik senkron mod başlatılıyor...")
            run_synchronous_loop()
            break
        else:
            logger.warning("Geçersiz seçim. Lütfen 1 veya 2 girin.")

    logger.info("Sesli Asistan kapatıldı. İyi günler!")