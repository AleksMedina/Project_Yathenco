import os
import asyncio
import logging
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket
from concurrent.futures import ThreadPoolExecutor

#  Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotionSystem")

app = FastAPI()

#  Глобальная переменная для REST API
last_analysis_result = {"success": False, "error": "No data yet", "faces": []}

#  Конфигурация путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YUNET_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
FER_PATH = os.path.join(BASE_DIR, "emotion-ferplus-8.onnx")

if not os.path.exists(YUNET_PATH) or not os.path.exists(FER_PATH):
    logger.error("Модели .onnx не найдены в директории скрипта!")
    raise FileNotFoundError("Убедитесь, что файлы моделей лежат рядом с main.py")

# Пул потоков для изоляции вызовов нейросетей
executor = ThreadPoolExecutor(max_workers=4)

# Инициализация моделей
detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (640, 480), score_threshold=0.6, nms_threshold=0.3)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
ort_session = ort.InferenceSession(FER_PATH, providers=['CPUExecutionProvider'], sess_options=opts)

# Инициализация CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

emotions_list = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']
INPUT_NAME = ort_session.get_inputs()[0].name

# Коэффициенты EMA
ALPHA_PROBS = 0.9
ALPHA_BOX = 0.9

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0


def process_frame_sync(frame_bytes, client_state):
    if len(frame_bytes) < 1024:
        return {"success": False, "error": "Invalid frame payload"}

    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Frame decode failed"}

        _, faces = detector.detect(frame)
        results = []
        current_frame_faces = {}

        if faces is not None:
            for face in faces:
                raw_box = face[:4].astype(np.float32)
                x, y, w, h = raw_box.astype(np.int32)

                # Границы
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                if x2 <= x1 + 10 or y2 <= y1 + 10: continue

                # Трекинг
                matched_id = None
                best_iou = 0.3
                for face_id, state in client_state.items():
                    iou = get_iou(raw_box, state['box'])
                    if iou > best_iou:
                        best_iou = iou
                        matched_id = face_id

                if matched_id is None:
                    matched_id = os.urandom(4).hex()
                    smoothed_box = raw_box
                    smoothed_probs = None
                else:
                    smoothed_box = ALPHA_BOX * raw_box + (1 - ALPHA_BOX) * client_state[matched_id]['box']
                    smoothed_probs = client_state[matched_id]['probs']

                # Подготовка лица с CLAHE
                face_crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                face_equalized = clahe.apply(face_crop)
                face_resized = cv2.resize(face_equalized, (64, 64), interpolation=cv2.INTER_LINEAR)
                input_tensor = np.reshape(face_resized.astype(np.float32), (1, 1, 64, 64))

                # Инференс
                raw_outputs = ort_session.run(None, {INPUT_NAME: input_tensor})[0][0]
                probs = np.exp(raw_outputs - np.max(raw_outputs))
                probs /= probs.sum()

                if smoothed_probs is not None:
                    probs = ALPHA_PROBS * probs + (1 - ALPHA_PROBS) * smoothed_probs

                current_frame_faces[matched_id] = {'box': smoothed_box, 'probs': probs}
                max_idx = np.argmax(probs)

                results.append({
                    "id": matched_id,
                    "emotion": emotions_list[max_idx],
                    "confidence": int(probs[max_idx] * 100),
                    "box": [int(smoothed_box[0]), int(smoothed_box[1]), int(smoothed_box[2]), int(smoothed_box[3])],
                    "all_emotions": {em: int(p * 100) for em, p in zip(emotions_list, probs)}
                })

        client_state.clear()
        client_state.update(current_frame_faces)
        return {"success": True, "faces": results}

    except Exception as e:
        return {"success": False, "error": str(e)}


# REST API
@app.get("/api/v1/emotional")
async def get_emotional_data():
    """Возвращает последний результат анализа в формате JSON"""
    return last_analysis_result


# WebSocket API
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global last_analysis_result
    await websocket.accept()
    frame_queue = asyncio.Queue(maxsize=1)
    client_state = {}

    async def receive_frames():
        try:
            while True:
                data = await websocket.receive_bytes()
                if frame_queue.full(): frame_queue.get_nowait()
                await frame_queue.put(data)
        except:
            pass

    async def process_and_send():
        global last_analysis_result
        try:
            while True:
                frame_bytes = await frame_queue.get()
                result = await asyncio.get_running_loop().run_in_executor(
                    executor, process_frame_sync, frame_bytes, client_state
                )

                # Сохраняем результат для REST API
                last_analysis_result = result

                await websocket.send_json(result)
        except:
            pass

    producer = asyncio.create_task(receive_frames())
    consumer = asyncio.create_task(process_and_send())
    try:
        await asyncio.gather(producer, consumer)
    finally:
        producer.cancel()
        consumer.cancel()