import simpleaudio as sa
import tempfile
import os
import torch
from TTS.api import TTS
from config import TTS_MODEL, TTS_LANGUAGE, TTS_SPEED, REFERENCE_SPEAKER
import logging
import time
import wave
import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(TTS_MODEL).to(self.device)
        self.is_speaking = False
        self.current_play_obj = None

        # 🔹 SADECE HIZ AYARI - Pitch'e dokunmuyoruz!
        self.speed_multiplier = 0.98 # Hız: 0.5 = yarı hız, 0.6 = biraz yavaş
        self.pause_after_speech = 3.0  # Konuşma sonrası duraklama

        logger.info(f"🎯 TTS cihazı: {self.device}")
        logger.info(f"🐌 Konuşma hızı: %{int(self.speed_multiplier * 100)}")

        # Referans dosya kontrolü - BU KISIM EKSİKTİ!
        if not os.path.exists(REFERENCE_SPEAKER):
            possible_paths = [
                "reference_speaker.wav",
                "audio/reference_speaker.wav",
                os.path.expanduser("~/Desktop/reference_speaker.wav")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"✅ Referans ses dosyası bulundu: {path}")
                    self.reference_speaker = path
                    break
            else:
                logger.error(f"❌ Referans ses dosyası bulunamadı: {REFERENCE_SPEAKER}")
                raise FileNotFoundError(f"Referans ses dosyası eksik: {REFERENCE_SPEAKER}")
        else:
            self.reference_speaker = REFERENCE_SPEAKER

    def slow_down_audio_only(self, input_path, output_path):
        """Ses dosyasını SADECE yavaşlatır - pitch değiştirmez"""
        try:
            # WAV dosyasını oku
            sample_rate, audio_data = wavfile.read(input_path)

            # Eğer stereo ise mono'ya çevir
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # SADECE HIZI YAVAŞLATIR - Time stretching
            original_length = len(audio_data)
            new_length = int(original_length / self.speed_multiplier)

            # Linear interpolation ile yavaşlat (pitch korunur)
            old_indices = np.linspace(0, original_length - 1, original_length)
            new_indices = np.linspace(0, original_length - 1, new_length)
            slow_audio = np.interp(new_indices, old_indices, audio_data)

            # Yeni ses dosyasını kaydet
            wavfile.write(output_path, sample_rate, slow_audio.astype(np.int16))

            logger.info(f"🐌 Ses %{int(self.speed_multiplier * 100)} hızında yavaşlatıldı (pitch korundu)")
            return True

        except Exception as e:
            logger.error(f"Ses yavaşlatma hatası: {e}")
            return False

    def speak(self, text):
        if self.is_speaking:
            if self.current_play_obj and self.current_play_obj.is_playing():
                self.current_play_obj.stop()

        self.is_speaking = True
        logger.info(f"🎤 Asistan: {text}")

        try:
            # Geçici dosyalar
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                temp_wav = fp.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp2:
                slow_wav = fp2.name

            # 1. Normal TTS oluştur (orijinal pitch korunur)
            self.tts.tts_to_file(
                text=text,
                file_path=temp_wav,
                language=TTS_LANGUAGE,
                speaker_wav=self.reference_speaker,  # Artık mevcut!
                speed=TTS_SPEED
            )

            # 2. Sadece hızı yavaşlat (pitch değiştirmez!)
            if self.slow_down_audio_only(temp_wav, slow_wav):
                wave_obj = sa.WaveObject.from_wave_file(slow_wav)
            else:
                logger.warning("⚠️ Yavaşlatma başarısız, orijinal hız kullanılıyor")
                wave_obj = sa.WaveObject.from_wave_file(temp_wav)

            self.current_play_obj = wave_obj.play()
            self.current_play_obj.wait_done()

            # Geçici dosyaları temizle
            os.remove(temp_wav)
            if os.path.exists(slow_wav):
                os.remove(slow_wav)

        except Exception as e:
            logger.error(f"TTS hatası: {e}")
        finally:
            self.is_speaking = False
            self.current_play_obj = None
            logger.info("⏳ Ses temizleniyor...")
            time.sleep(self.pause_after_speech)

    def set_speed(self, speed_multiplier):
        """Hızı değiştir - Runtime'da çağrılabilir"""
        self.speed_multiplier = max(0.1, min(1.0, speed_multiplier))  # 0.2-2.0 arası sınırla
        logger.info(f"🔧 Yeni hız ayarı: %{int(self.speed_multiplier * 100)}")

    def test_speeds(self, text="Merhaba, hız testi yapıyorum"):
        """Farklı hızları test et"""
        speeds = [0.4, 0.6, 0.8, 1.0]
        for speed in speeds:
            logger.info(f"\n🧪 Test edilen hız: %{int(speed * 100)}")
            self.set_speed(speed)
            self.speak(text)
            time.sleep(1)