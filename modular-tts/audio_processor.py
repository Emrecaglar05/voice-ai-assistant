import numpy as np
import speech_recognition as sr
from scipy.signal import butter, filtfilt
import noisereduce as nr
import logging
import whisper
import tempfile
import os

from config import SAMPLE_RATE, CHUNK_SIZE

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        self.recognizer = sr.Recognizer()

        # 🔹 Hassas ayarlar
        self.recognizer.energy_threshold = 200
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8

        self.mic = sr.Microphone(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)

        # 🔹 Whisper modelini yükle
        try:
            logger.info("🔄 Whisper modeli yükleniyor...")
            # Model seçenekleri: tiny, base, small, medium, large
            # tiny: en hızlı, base: dengeli, small: iyi doğruluk
            self.whisper_model = whisper.load_model("base")  # veya "small"
            logger.info("✅ Whisper modeli başarıyla yüklendi!")
            self.use_whisper = True
        except Exception as e:
            logger.error(f"❌ Whisper modeli yüklenemedi: {e}")
            logger.warning("⚠️ Whisper kullanılamıyor!")
            self.use_whisper = False

    # 🔹 Preprocessing fonksiyonu
    def preprocess_audio(self, audio_data):
        try:
            audio = np.frombuffer(audio_data.get_raw_data(), np.int16).astype(np.float32)

            # Bandpass filtre
            nyquist = self.sample_rate * 0.5
            b, a = butter(4, [300 / nyquist, 3400 / nyquist], btype='band')
            audio = filtfilt(b, a, audio)

            # Gürültü azaltma
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate, prop_decrease=0.8, stationary=True)
            audio = nr.reduce_noise(y=audio, sr=self.sample_rate, prop_decrease=0.6, stationary=False)

            # Normalizasyon
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8

            # Int16 formatına çevir
            audio_int16 = (audio * 32767).astype(np.int16)
            return sr.AudioData(audio_int16.tobytes(), audio_data.sample_rate, audio_data.sample_width)

        except Exception as e:
            logger.error(f"Audio preprocessing hatası: {e}")
            return audio_data

    # 🔹 Whisper ile ses tanıma
    def process_audio_whisper(self, audio):
        try:
            clean_audio = self.preprocess_audio(audio)

            # Geçici WAV dosyası oluştur
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                tmp_file.write(clean_audio.get_wav_data())

            # Whisper ile transcribe et
            result = self.whisper_model.transcribe(
                tmp_filename,
                language="tr",  # Türkçe
                fp16=False,  # CPU için False, GPU için True
                verbose=False
            )

            # Geçici dosyayı sil
            os.unlink(tmp_filename)

            user_input = result["text"].strip()

            if user_input:
                # Whisper'ın kendi güven skorunu kullan
                # segments varsa ortalama güven skorunu al
                if "segments" in result and result["segments"]:
                    confidences = [seg.get("no_speech_prob", 0) for seg in result["segments"]]
                    confidence = 1.0 - (sum(confidences) / len(confidences))
                else:
                    confidence = 0.85  # Varsayılan yüksek güven

                logger.info(f"👤 Kullanıcı: {user_input} (Güven: {confidence:.2f})")
                return user_input if confidence >= 0.3 else "CONFIDENCE_TOO_LOW"

            return None

        except Exception as e:
            logger.error(f"Whisper ses işleme hatası: {e}")
            return None

    # 🔹 Ses tanıma (Whisper kullan)
    def process_audio(self, audio):
        if self.use_whisper:
            return self.process_audio_whisper(audio)
        else:
            logger.error("❌ Whisper mevcut değil!")
            return None

    # 🔹 Dinleme fonksiyonu
    def listen(self, is_speaking=False, listening_active=True):
        if is_speaking or not listening_active:
            return None

        logger.info("🎧 Dinliyorum...")
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info(f"🔊 Gürültü eşiği: {self.recognizer.energy_threshold}")

                audio = self.recognizer.listen(
                    source,
                    timeout=10,
                    phrase_time_limit=20
                )
                logger.info("✅ Ses yakalandı!")
                return audio

        except sr.WaitTimeoutError:
            logger.info("⏳ Ses gelmedi (timeout)")
            return None
        except Exception as e:
            logger.error(f"Dinleme hatası: {e}")
            return None

    # 🔹 Mikrofon testi
    def test_microphone(self):
        with self.mic as source:
            print("🎧 Lütfen konuşun...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            try:
                if self.use_whisper:
                    text = self.process_audio_whisper(audio)
                    print(f"✅ Algılanan ses: {text}")
                else:
                    print("❌ Whisper mevcut değil")
            except Exception as e:
                print(f"❌ Hata: {e}")

    # 🔹 Debug fonksiyonu
    def debug_microphone(self):
        """Mikrofon sorunlarını tespit et"""
        logger.info("🔍 Mikrofon debug başlıyor...")

        try:
            with self.mic as source:
                logger.info("✅ Mikrofon erişimi başarılı")

                logger.info("📊 3 saniye ortam gürültüsü ölçülüyor...")
                self.recognizer.adjust_for_ambient_noise(source, duration=3)
                logger.info(f"🔊 Tespit edilen gürültü seviyesi: {self.recognizer.energy_threshold}")

                logger.info("🎤 Şimdi konuşun (10 saniye)...")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                logger.info("✅ Ses başarıyla kaydedildi!")

                # Ham veriyi kontrol et
                raw_data = np.frombuffer(audio.get_raw_data(), np.int16)
                logger.info(
                    f"📈 Ses seviyesi - Min: {raw_data.min()}, Max: {raw_data.max()}, Mean: {raw_data.mean():.2f}")

                if abs(raw_data.max()) < 100:
                    logger.warning("⚠️ Ses seviyesi çok düşük! Mikrofon sesini yükseltmeyi deneyin.")

                # Whisper ile test
                if self.use_whisper:
                    result = self.process_audio_whisper(audio)
                    logger.info(f"🎯 Whisper sonucu: {result}")

                return True

        except Exception as e:
            logger.error(f"❌ Mikrofon hatası: {e}")
            return False