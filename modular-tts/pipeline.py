import asyncio
from concurrent.futures import ThreadPoolExecutor
from config import MAX_WORKERS
import logging
from audio_processor import AudioProcessor
from ai_response import get_ai_response
from tts_engine import TTSEngine

logger = logging.getLogger(__name__)

class AsyncPipelineManager:
    def __init__(self, max_workers=MAX_WORKERS):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.audio_queue = asyncio.Queue(maxsize=5)
        self.ai_queue = asyncio.Queue(maxsize=5)
        self.tts_queue = asyncio.Queue(maxsize=5)
        self.pipeline_active = True
        self.processing_stats = {'audio_processed': 0, 'ai_responses': 0, 'tts_generated': 0}

    async def start_pipeline(self):
        logger.info("🚀 Asenkron pipeline başlatılıyor...")
        pipeline_tasks = [
            asyncio.create_task(self.audio_processor_worker()),
            asyncio.create_task(self.ai_response_worker()),
            asyncio.create_task(self.tts_worker())
        ]
        await asyncio.gather(*pipeline_tasks)

    async def audio_processor_worker(self):
        audio_processor = AudioProcessor()
        while self.pipeline_active:
            try:
                audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                loop = asyncio.get_event_loop()
                processed_text = await loop.run_in_executor(
                    self.executor, audio_processor.process_audio, audio_data
                )
                if processed_text:
                    await self.ai_queue.put(processed_text)
                    self.processing_stats['audio_processed'] += 1
                    logger.info(f"🎙️ Ses işlendi: {processed_text[:50]}...")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Ses işleme işçisi hatası: {e}")
                await asyncio.sleep(0.1)

    async def ai_response_worker(self):
        while self.pipeline_active:
            try:
                user_text = await asyncio.wait_for(self.ai_queue.get(), timeout=1.0)
                loop = asyncio.get_event_loop()
                ai_response = await loop.run_in_executor(
                    self.executor, get_ai_response, user_text
                )
                if ai_response:
                    await self.tts_queue.put(ai_response)
                    self.processing_stats['ai_responses'] += 1
                    logger.info(f"🤖 AI yanıt hazır: {ai_response[:50]}...")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"AI işçisi hatası: {e}")
                await asyncio.sleep(0.1)

    async def tts_worker(self):
        tts_engine = TTSEngine()
        while self.pipeline_active:
            try:
                text_to_speak = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, tts_engine.speak, text_to_speak)
                self.processing_stats['tts_generated'] += 1
                logger.info("🎵 TTS tamamlandı")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"TTS işçisi hatası: {e}")
                await asyncio.sleep(0.1)

    async def add_audio_to_pipeline(self, audio_data):
        try:
            if self.audio_queue.full():
                self.audio_queue.get_nowait()
            await self.audio_queue.put(audio_data)
            logger.info("🎧 Ses pipeline'a eklendi")
        except Exception as e:
            logger.error(f"Pipeline ekleme hatası: {e}")

    def stop_pipeline(self):
        self.pipeline_active = False
        self.executor.shutdown(wait=True)
        logger.info("🛑 Asenkron pipeline durduruldu")

    def get_pipeline_stats(self):
        return {
            "queue_sizes": {
                "audio": self.audio_queue.qsize(),
                "ai": self.ai_queue.qsize(),
                "tts": self.tts_queue.qsize()
            },
            "processed": self.processing_stats
        }