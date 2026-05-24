import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")

    YUNET_PATH: str = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
    FER_PATH: str = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")

    SCORE_THRESHOLD: float = 0.6
    NMS_THRESHOLD: float = 0.3

    # СМЕЩЕНИЯ ЛОГИТОВ (Logit Biases)
    # Порядок: neutral, happy, surprise, sad, angry, disgust, fear, contempt
    EMOTION_BIASES: list[float] = [
        -0.8,  # neutral  (Спокойствие)
        1.0,  # happy    (Радость)
        1.2,  # surprise (Удивление)
        -0.2,  # sad      (Грусть / Печаль)
        1.5,  # angry    (Злость / Гнев)
        1.8,  # disgust  (Отвращение)
        3.5,  # fear     (Страх)
        2.96  # contempt (Презрение)
    ]

    ALPHA_PROBS: float = 0.85
    ALPHA_BOX: float = 0.85


config = Settings()