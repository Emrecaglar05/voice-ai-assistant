import asyncio
import logging
import time
from audio_processor import AudioProcessor
from pipeline import AsyncPipelineManager
from config import GOOGLE_API_KEY, EXIT_WORDS
from tts_engine import TTSEngine
from cache_manager import SmartSemanticCache
from conversation import ConversationMemory
from ai_response import get_ai_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.pipeline_manager = AsyncPipelineManager()
        self.tts_engine = TTSEngine()
        self.smart_cache = SmartSemanticCache()
        self.conversation_memory = ConversationMemory()
        self.is_speaking = False
        self.listening_active = True

    def show_stats(self):
        logger.info("\n📊 Sistem Durumu:")
        logger.info(f"   Konuşma geçmişi: {self.conversation_memory.get_stats()['history_size']} exchange")
        cache_stats = self.smart_cache.get_cache_stats()
        logger.info(f"   Önbellek hit oranı: {cache_stats['hit_rate']}")
        logger.info(f"   Önbellek boyutu: {cache_stats['cache_size']}")
        logger.info(f"   Semantik önbellek: {cache_stats['semantic_cache_size']}")
        logger.info(f"   Semantik eşleşme: {cache_stats['semantic_hits']}")
        pipeline_stats = self.pipeline_manager.get_pipeline_stats()
        logger.info(f"   Pipeline kuyrukları: Audio:{pipeline_stats['queue_sizes']['audio']} AI:{pipeline_stats['queue_sizes']['ai']} TTS:{pipeline_stats['queue_sizes']['tts']}")
        logger.info(f"   İşlenen: Ses:{pipeline_stats['processed']['audio_processed']} AI:{pipeline_stats['processed']['ai_responses']} TTS:{pipeline_stats['processed']['tts_generated']}")
        logger.info(f"   Mevcut konu: {self.conversation_memory.get_stats()['current_topic']}")

    def initialize(self):
        logger.info(" Geliştirilmiş Sesli Asistan başlatılıyor...")
        if not GOOGLE_API_KEY:
            raise ValueError(" GOOGLE_API_KEY bulunamadı!")
        logger.info("🔐 API key güvenli şekilde yüklendi")
        with self.audio_processor.mic as source:
            self.audio_processor.recognizer.adjust_for_ambient_noise(source, duration=3)
        logger.info(f" Kalibrasyon tamamlandı. Gürültü eşiği: {self.audio_processor.recognizer.energy_threshold}")
        self.smart_cache.preload_common_responses()
        self.tts_engine.speak("Merhaba evlat")


    async def async_main_loop(self):
        logger.info(" Asenkron ana döngü başlatılıyor...")
        pipeline_task = asyncio.create_task(self.pipeline_manager.start_pipeline())
        consecutive_errors = 0
        max_consecutive_errors = 5
        try:
            while True:
                try:
                    if not self.listening_active or self.is_speaking:
                        await asyncio.sleep(0.2)
                        continue
                    audio = await asyncio.get_event_loop().run_in_executor(
                        self.pipeline_manager.executor,
                        lambda: self.audio_processor.listen(self.is_speaking, self.listening_active)
                    )
                    if audio:
                        await self.pipeline_manager.add_audio_to_pipeline(audio)
                        consecutive_errors = 0
                    else:
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Ana döngü hatası: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("🚨 Çok fazla hata! Sistem yeniden başlatılıyor...")
                        consecutive_errors = 0
                        await asyncio.sleep(2)
        except KeyboardInterrupt:
            logger.info("🛑 Kullanıcı tarafından durduruldu")
        finally:
            self.pipeline_manager.stop_pipeline()
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                logger.info("Pipeline durduruldu")

    def sync_main_loop(self):
        logger.info("🔄 Klasik senkron mod başlatılıyor...")
        consecutive_errors = 0
        max_consecutive_errors = 5
        while True:
            try:
                if not self.listening_active or self.is_speaking:
                    time.sleep(0.2)
                    continue

                # Düzeltilmiş çağrı - parametreleri geç
                audio = self.audio_processor.listen(self.is_speaking, self.listening_active)
                if not audio:
                    time.sleep(0.1)
                    continue

                # Ses verisini işle
                user_input = self.audio_processor.process_audio(audio)
                if user_input and user_input != "CONFIDENCE_TOO_LOW":
                    consecutive_errors = 0
                    if any(word in user_input.lower() for word in EXIT_WORDS):
                        self.tts_engine.speak("Görüşürüz! İyi günler dilerim.")
                        self.show_stats()
                        break
                    response = get_ai_response(user_input, self.smart_cache, self.conversation_memory)
                    self.tts_engine.speak(response)
                elif user_input == "CONFIDENCE_TOO_LOW":
                    self.tts_engine.speak("Seni tam anlayamadım, biraz daha yüksek sesle söyler misin?")
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.tts_engine.speak("Ses kalitesi düşük. Mikrofona daha yakın konuş!")
                        consecutive_errors = 0
                if self.conversation_memory.get_stats()['history_size'] % 10 == 0:
                    self.show_stats()
            except KeyboardInterrupt:
                logger.info("🛑 Kullanıcı tarafından durduruldu")
                self.show_stats()
                break
            except Exception as e:
                logger.error(f"Ana döngü hatası: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("🚨 Çok fazla hata! Sistem yeniden başlatılıyor...")
                    consecutive_errors = 0
                    time.sleep(2)

    def run(self):
        self.initialize()
        print("\n🚀 Çalışma modu seçin:")
        print("1. Asenkron Pipeline Modu (Önerilen)")
        print("2. Klasik Senkron Mod")
        while True:
            try:
                mode_choice = input("Seçiminiz (1/2): ").strip()
                if mode_choice == "1":
                    logger.info("🚀 Asenkron pipeline modu başlatılıyor...")
                    asyncio.run(self.async_main_loop())
                    break
                elif mode_choice == "2":
                    self.sync_main_loop()
                    break
                else:
                    print("❌ Geçersiz seçim! 1 veya 2 girin.")
            except KeyboardInterrupt:
                logger.info("🛑 Program durduruldu")
                break
        logger.info("👋 Sesli Asistan kapatıldı!")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()