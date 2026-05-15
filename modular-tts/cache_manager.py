from collections import deque
import re
import time
from difflib import SequenceMatcher
from config import CACHE_MAX_SIZE, SIMILARITY_THRESHOLD
import logging

logger = logging.getLogger(__name__)

class SmartSemanticCache:
    def __init__(self, max_size=CACHE_MAX_SIZE, similarity_threshold=SIMILARITY_THRESHOLD):
        self.cache = {}
        self.semantic_cache = {}
        self.access_count = {}
        self.last_access = {}
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.stats = {'hits': 0, 'misses': 0, 'semantic_hits': 0, 'cache_size': 0}
        self.predefined_patterns = {
            'greeting': ['merhaba', 'selam', 'nasılsın', 'naber', 'hey', 'hi'],
            'thanks': ['teşekkür', 'sağ ol', 'thank you', 'merci'],
            'goodbye': ['görüşürüz', 'hoşça kal', 'bay bay', 'çıkış', 'bitir'],
            'general_help': ['yardım', 'ne yapabiliriz', 'neler var']
        }

    def _generate_cache_key(self, text):
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        stop_words = {'bir', 'bu', 'şu', 'o', 'ben', 'sen', 've', 'ile', 'için', 'gibi', 'mi', 'mı'}
        key_words = [w for w in words if len(w) > 2 and w not in stop_words]
        return ' '.join(sorted(key_words[:4]))

    def _calculate_similarity(self, text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_score = intersection / union if union > 0 else 0.0
        sequence_score = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return (jaccard_score * 0.6) + (sequence_score * 0.4)

    def _find_pattern_match(self, user_input):
        user_lower = user_input.lower()
        for pattern_name, keywords in self.predefined_patterns.items():
            if any(keyword in user_lower for keyword in keywords):
                pattern_key = f"pattern_{pattern_name}"
                if pattern_key in self.cache:
                    logger.info(f"🎯 Pattern eşleşmesi bulundu: {pattern_name}")
                    return self.cache[pattern_key]
        return None

    def get(self, user_input):
        cache_key = self._generate_cache_key(user_input)
        if cache_key in self.cache:
            self.stats['hits'] += 1
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            self.last_access[cache_key] = time.time()
            logger.info(f"💾 Önbellek HIT (tam): {cache_key}")
            return self.cache[cache_key]

        pattern_result = self._find_pattern_match(user_input)
        if pattern_result:
            self.stats['hits'] += 1
            return pattern_result

        best_score = 0.0
        best_match = None
        best_key = None
        for cached_input, response in self.semantic_cache.items():
            similarity = self._calculate_similarity(user_input, cached_input)
            if similarity > best_score and similarity >= self.similarity_threshold:
                best_score = similarity
                best_match = response
                best_key = cached_input

        if best_match:
            self.stats['semantic_hits'] += 1
            logger.info(f"🎯 Semantik eşleşme bulundu: {best_key} (skor: {best_score:.2f})")
            return best_match

        self.stats['misses'] += 1
        return None

    def set(self, user_input, response):
        cache_key = self._generate_cache_key(user_input)
        if len(self.cache) >= self.max_size:
            self._evict_least_valuable()
        self.cache[cache_key] = response
        self.semantic_cache[user_input] = response
        self.access_count[cache_key] = 1
        self.last_access[cache_key] = time.time()
        self.stats['cache_size'] = len(self.cache)
        logger.info(f"💾 Önbelleğe kaydedildi: {cache_key}")

    def _evict_least_valuable(self):
        if not self.cache:
            return
        current_time = time.time()
        scores = {}
        for key in self.cache.keys():
            frequency_score = self.access_count.get(key, 1)
            recency_score = current_time - self.last_access.get(key, 0)
            scores[key] = frequency_score - (recency_score / 3600)

        least_valuable = min(scores.items(), key=lambda x: x[1])
        evicted_key = least_valuable[0]

        del self.cache[evicted_key]
        if evicted_key in self.access_count:
            del self.access_count[evicted_key]
        if evicted_key in self.last_access:
            del self.last_access[evicted_key]

        to_remove = []
        for semantic_key, response in self.semantic_cache.items():
            if self._generate_cache_key(semantic_key) == evicted_key:
                to_remove.append(semantic_key)

        for key in to_remove:
            del self.semantic_cache[key]

        logger.info(f"🗑️ Önbellekten silindi: {evicted_key} (skor: {least_valuable[1]:.2f})")

    def get_cache_stats(self):
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        return {
            'hit_rate': f"{hit_rate:.1f}%",
            'total_hits': self.stats['hits'],
            'semantic_hits': self.stats['semantic_hits'],
            'cache_size': len(self.cache),
            'semantic_cache_size': len(self.semantic_cache)
        }

    def preload_common_responses(self):
        common_qa_pairs = [
            ("merhaba nasılsın", "Merhaba! İyiyim, teşekkür ederim. Sen nasılsın?"),
            ("nasılsın", "İyiyim, teşekkürler! Sen nasılsın?"),
            ("teşekkürler", "Rica ederim! Başka bir konuda yardımcı olabilirim."),
            ("yardım et", "Tabii ki! Hangi konuda yardıma ihtiyacın var?"),
            ("görüşürüz", "Görüşürüz! İyi günler dilerim."),
            ("bilmiyorum", "Anladım. Başka nasıl yardımcı olabilirim?")
        ]
        for question, answer in common_qa_pairs:
            self.set(question, answer)
        logger.info(f"✅ {len(common_qa_pairs)} yaygın yanıt önbelleğe yüklendi")