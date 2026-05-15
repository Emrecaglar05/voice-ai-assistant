# tts_test_safe_all.py
import torch
import inspect
import b  # otomatik whitelist

from TTS.tts import configs as tts_configs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.xtts_config import (
    XttsConfig,
    XttsAudioConfig
)

# -------------------------
# Coqui TTS içindeki tüm config sınıflarını topla
# -------------------------
safe_classes = [BaseAudioConfig, XttsConfig, XttsAudioConfig]

# tts.configs altındaki tüm class'ları ekle
for name, obj in inspect.getmembers(tts_configs):
    if inspect.isclass(obj):
        safe_classes.append(obj)

# Tekrarlı olanları ayıkla
safe_classes = list(set(safe_classes))

# -------------------------
# PyTorch güvenli global listesine ekle
# -------------------------
torch.serialization.add_safe_globals(safe_classes)

# -------------------------
# Modeli yükle
# -------------------------
from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# -------------------------
# Örnek metin + referans ses
# -------------------------
text ="evet ben ikibindokuzdan beri belediyedeyim önce başkan yardımcılığı yaptım ikibinondörtte bandırma belediye başkanıydım şimdi tekrar ikinci kez başkan seçildim ama böyle bir dönem görmedim yani böylesine",

output_file = "output_speech.wav"

tts.tts_to_file(
    text=text,
    file_path=output_file,
    speaker_wav="C:/Desktop/Dursun Mirza Ses kayıt/2.wav",  # 🔹 buraya kendi referans ses dosyanı koy
    language="tr"
)

print(f"Ses dosyası oluşturuldu: {output_file}")
