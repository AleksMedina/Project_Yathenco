import os
import asyncio
import logging
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotionSystem")

app = FastAPI()

# --- Конфигурация ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
FER_PATH = os.path.join(BASE_DIR, "emotion-ferplus-8.onnx")

# --- Инициализация ---
if not os.path.exists(YUNET_PATH) or not os.path.exists(FER_PATH):
    logger.error("Модели .onnx не найдены!")
    raise FileNotFoundError("Модели не найдены.")

detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (0, 0), score_threshold=0.6, nms_threshold=0.3)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
ort_session = ort.InferenceSession(FER_PATH, sess_options=opts)

emotions_list = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']
INPUT_NAME = ort_session.get_inputs()[0].name

# Коэффициенты сглаживания
ALPHA_PROBS = 0.2
ALPHA_BOX = 0.15


def process_frame(frame_bytes, smoothed_probs, last_smoothed_box):
    """
    Обработка кадра с защитой от ошибок.
    """
    try:
        # 1. Декодирование
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return False, "neutral", 0, {e: 0 for e in emotions_list}, None, smoothed_probs, last_smoothed_box

        h_frame, w_frame, _ = frame.shape
        detector.setInputSize((w_frame, h_frame))
        _, faces = detector.detect(frame)

        if faces is not None and len(faces) > 0:
            raw_box = faces[0][:4].astype(np.float32)

            # EMA Сглаживание рамки
            smoothed_box = raw_box if last_smoothed_box is None else (
                        ALPHA_BOX * raw_box + (1 - ALPHA_BOX) * last_smoothed_box)
            x, y, w, h = smoothed_box.astype(np.int32)

            # --- ВАЖНАЯ ПРОВЕРКА ROI ---
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_frame, x + w), min(h_frame, y + h)

            if x2 <= x1 + 10 or y2 <= y1 + 10:
                return False, "neutral", 0, {e: 0 for e in emotions_list}, None, smoothed_probs, last_smoothed_box

            # Кроп и ресайз
            face_crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_LINEAR)
            input_tensor = np.reshape(face_resized.astype(np.float32), (1, 1, 64, 64))

            # Инференс
            raw_outputs = ort_session.run(None, {INPUT_NAME: input_tensor})[0][0]
            # Softmax
            probs = np.exp(raw_outputs - np.max(raw_outputs))
            probs /= probs.sum()

            smoothed_probs = probs if smoothed_probs is None else (
                        ALPHA_PROBS * probs + (1 - ALPHA_PROBS) * smoothed_probs)

            max_idx = np.argmax(smoothed_probs)
            all_emotions = {em: int(p * 100) for em, p in zip(emotions_list, smoothed_probs)}

            return True, emotions_list[max_idx], int(smoothed_probs[max_idx] * 100), all_emotions, [int(x), int(y),
                                                                                                    int(w),
                                                                                                    int(h)], smoothed_probs, smoothed_box

    except Exception as e:
        logger.error(f"Ошибка при обработке кадра: {e}")

    return False, "neutral", 0, {e: 0 for e in emotions_list}, None, smoothed_probs, last_smoothed_box


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Клиент подключился")
    probs = None
    box = None
    try:
        while True:
            data = await websocket.receive_bytes()
            # Выполнение в потоке, чтобы не блокировать event loop
            res = await asyncio.to_thread(process_frame, data, probs, box)

            # Распаковка результатов
            success, emotion, conf, all_emotions, box_coords, probs, box = res

            await websocket.send_json({
                "success": success,
                "emotion": emotion,
                "confidence": conf,
                "all_emotions": all_emotions,
                "box": box_coords
            })
    except WebSocketDisconnect:
        logger.info("Клиент отключился")
    except Exception as e:
        logger.error(f"WebSocket ошибка: {e}")