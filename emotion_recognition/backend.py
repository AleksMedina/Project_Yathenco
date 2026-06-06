import os
import asyncio
import logging
import cv2
import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket
from concurrent.futures import ThreadPoolExecutor
from core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotionSystem")

app = FastAPI()
last_analysis_result = {"success": False, "error": "No data yet", "faces": []}

# Проверка моделей
if not os.path.exists(config.YUNET_PATH) or not os.path.exists(config.FER_PATH):
    raise FileNotFoundError("Модели ONNX не найдены в папке models/")

executor = ThreadPoolExecutor(max_workers=4)

detector = cv2.FaceDetectorYN.create(
    config.YUNET_PATH, "", (640, 480),
    score_threshold=config.SCORE_THRESHOLD,
    nms_threshold=config.NMS_THRESHOLD
)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
ort_session = ort.InferenceSession(config.FER_PATH, providers=['CPUExecutionProvider'], sess_options=opts)
INPUT_NAME = ort_session.get_inputs()[0].name
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

emotions_list = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']
biases = np.array(config.EMOTION_BIASES)


def get_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0


def process_frame_sync(frame_bytes, client_state):
    start_time = time.time()
    if len(frame_bytes) < 1024:
        return {"success": False, "error": "Invalid payload"}

    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return {"success": False, "error": "Decode failed"}

        _, faces = detector.detect(frame)
        results = []
        current_frame_faces = {}

        if faces is not None:
            for face in faces:
                raw_box = face[:4].astype(np.float32)
                x, y, w, h = raw_box.astype(np.int32)
                x1, y1, x2, y2 = max(0, x), max(0, y), min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                if x2 <= x1 + 10 or y2 <= y1 + 10: continue

                matched_id, best_iou = None, 0.3
                for face_id, state in client_state.items():
                    iou = get_iou(raw_box, state['box'])
                    if iou > best_iou:
                        best_iou, matched_id = iou, face_id

                if matched_id is None:
                    matched_id = os.urandom(4).hex()
                    smoothed_box = raw_box
                    smoothed_probs = None
                else:
                    smoothed_box = config.ALPHA_BOX * raw_box + (1 - config.ALPHA_BOX) * client_state[matched_id]['box']
                    smoothed_probs = client_state[matched_id]['probs']

                face_crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                face_equalized = clahe.apply(face_crop)
                face_resized = cv2.resize(face_equalized, (64, 64))
                input_tensor = np.reshape(face_resized.astype(np.float32), (1, 1, 64, 64))

                raw_outputs = ort_session.run(None, {INPUT_NAME: input_tensor})[0][0]

                # --- 1. СЧИТАЕМ СЫРЫЕ (ЧЕСТНЫЕ) ВЕРОЯТНОСТИ ---
                raw_probs = np.exp(raw_outputs - np.max(raw_outputs))
                raw_probs /= raw_probs.sum()

                # Копируем базовые смещения из конфига
                dynamic_biases = biases.copy()

                # ИНДЕКСЫ ЭМОЦИЙ:
                # 0=neutral, 1=happy, 2=surprise, 3=sad, 4=angry, 5=disgust, 6=fear, 7=contempt

                # --- 2. УМНЫЕ КРОСС-ШТРАФЫ И ФИЛЬТРЫ ---

                # ПРАВИЛО А: Полностью вырезаем Презрение (Contempt - 7)
                dynamic_biases[7] = -100.0

                # ПРАВИЛО Б: Искусственный разгон Страха (Fear - 6)
                if raw_probs[6] > 0.03:
                    dynamic_biases[6] += 2.0

                # ПРАВИЛО В: Возврат Спокойствия (Neutral - 0)
                if np.max(raw_probs[1:]) < 0.20:
                    dynamic_biases[0] = 0.0

                # --- 3. ИТОГОВЫЙ РАСЧЕТ ---
                adjusted_logits = raw_outputs + dynamic_biases

                # Считаем классический Softmax от новых значений
                probs = np.exp(adjusted_logits - np.max(adjusted_logits))
                probs /= probs.sum()

                # Сглаживание EMA
                if smoothed_probs is not None:
                    probs = config.ALPHA_PROBS * probs + (1 - config.ALPHA_PROBS) * smoothed_probs

                current_frame_faces[matched_id] = {'box': smoothed_box, 'probs': probs}
                max_idx = np.argmax(probs)

                results.append({
                    "id": matched_id,
                    "emotion": emotions_list[max_idx],
                    "confidence": int(probs[max_idx] * 100),
                    "box": [int(smoothed_box[0]), int(smoothed_box[1]), int(smoothed_box[2]), int(smoothed_box[3])],
                    "all_emotions": {em: int(p * 100) for em, p in zip(emotions_list, probs) if em != 'contempt'}
                })

        client_state.clear()
        client_state.update(current_frame_faces)

        process_time_ms = int((time.time() - start_time) * 1000)
        return {"success": True, "faces": results, "process_time_ms": process_time_ms}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/v1/emotional")
async def get_emotional_data():
    return last_analysis_result


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