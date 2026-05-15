import os
from dotenv import load_dotenv

load_dotenv()

# API ve Çevre Ayarları
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
REFERENCE_SPEAKER = os.path.join(os.getcwd(), "audio", "reference_speaker.wav")

# Ses İşleme Ayarları
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 2.0
MIN_SPEECH_DURATION = 1.0

# Önbellek Ayarları
CACHE_MAX_SIZE = 150
SIMILARITY_THRESHOLD = 0.65

# Konuşma Hafızası Ayarları
MAX_HISTORY = 15

# Pipeline Ayarları
MAX_WORKERS = 4

# TTS Ayarları
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE = "tr"
TTS_SPEED = 1


# Temel Prompt
BASE_PROMPT = """
Ses Tanımı: "Türk erkek sesi, otoriter, net telaffuz, resmi konuşma tarzı, ölçülü tempo, kendinden emin ton, tarihi lider karakteri, 1920-1930 dönemi Türkçe telaffuzu, derin rezonans, ağırbaşlı konuşma kalıbı"
Eğitim Metinleri:
- "Yurtta sulh, cihanda sulh"
- "Egemenlik kayıtsız şartsız milletindir" 
- "Muhtaç olduğun kudret damarlarındaki asil kanda mevcuttur"
- "Türk milleti zekidir, Türk milleti çalışkandır"
- "Hayatta en hakiki mürşit ilimdir"
"""


# Soru Türü Anahtar Kelimeler
FACTUAL_KEYWORDS = ['nedir', 'kimdir', 'nerede', 'ne zaman', 'nasıl', 'kaç', 'kaçıncı', 'hangi', 'hangisi', 'ne kadar', 'kim', 'ne', 'nereye', 'nereden', 'bölüm', 'sayı', 'sayısı', 'tarih', 'yaş', 'yıl', 'hakkında', 'bilgi']
ADVICE_KEYWORDS = ['tavsiye', 'öneri', 'ne yapmalı', 'nasıl olmalı', 'yardım', 'ne yemeli', 'nasıl yapabilir', 'öner', 'tavsiye et', 'nasıl başlarım']
GREETING_KEYWORDS = ['merhaba', 'selam', 'nasılsın', 'iyi misin', 'hey']
EXIT_WORDS = ["quit", "çıkış", "kapat", "bitir", "son", "bye", "görüşürüz", "hoşça kal"]