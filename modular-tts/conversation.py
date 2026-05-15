from collections import deque
from datetime import datetime
from config import MAX_HISTORY
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self, max_history=MAX_HISTORY):
        self.history = deque(maxlen=max_history)
        self.context_keywords = {}
        self.current_topic = None

    def add_exchange(self, user_input, ai_response):
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'assistant': ai_response,
            'topic': self.extract_topic(user_input)
        }
        self.history.append(exchange)
        self.update_context(user_input)
        logger.info(f"💬 Konuşma kaydedildi: {user_input[:30]}...")

    def extract_topic(self, text):
        topics = {
            'eğitim': ['üniversite', 'okul', 'eğitim', 'ders', 'öğrenci', 'öğretmen', 'medipol'],
            'teknoloji': ['bilgisayar', 'yazılım', 'veri bilimi', 'programlama', 'teknoloji'],
            'eğlence': ['film', 'dizi', 'müzik', 'çizgi film', 'cedric', 'bölüm'],
            'sağlık': ['sağlık', 'hasta', 'ağrı', 'yorgun', 'vitamin'],
            'beslenme': ['yemek', 'beslenme', 'diyet', 'kilo', 'sebze'],
            'spor': ['spor', 'egzersiz', 'koşu', 'fitness'],
            'genel': ['nasıl', 'nedir', 'kim', 'ne']
        }
        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                self.current_topic = topic
                return topic
        return 'genel'

    def update_context(self, user_input):
        words = user_input.lower().split()
        for word in words:
            if len(word) > 3:
                self.context_keywords[word] = self.context_keywords.get(word, 0) + 1

    def get_context(self):
        if not self.history:
            return ""
        recent_context = [f"Önceki: {exchange['user'][:30]}..." for exchange in list(self.history)[-2:]]
        context = "\n".join(recent_context)
        return f"\nKısa geçmiş:\n{context}\nKonu: {self.current_topic or 'genel'}\n"

    def get_stats(self):
        return {'history_size': len(self.history), 'current_topic': self.current_topic or 'belirsiz'}