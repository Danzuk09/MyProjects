import os

# Установка переменной окружения для включения оптимизаций OneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import keras
import cv2
import mediapipe as mp
import numpy as np
import time

# Импорт функций для обработки данных
from functions import mediapipe_detection, extract_keypoints, draw_styled_landmarks_for_testing
from config_model import *

# Загрузка модели
model = keras.models.load_model('action.keras')
with open('actions.txt', encoding='utf-8') as f:
    actions_list = list(map(str.rstrip, f.readlines()))  # Изменено название переменной

ACTION_RECT_HEIGHT = 15  # Переименована константа
colors = [(16, 117, 245)] * len(actions_list)  # Изменено название переменной


def visualize_probabilities(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for index, probability in enumerate(res):  # Переименованы переменные
        cv2.rectangle(output_frame, (0, 40 + index * ACTION_RECT_HEIGHT),
                      (int(probability * 100), 40 + (index + 1) * ACTION_RECT_HEIGHT), colors[index], -1)
        cv2.putText(output_frame, actions[index], (0, 50 + index * ACTION_RECT_HEIGHT), cv2.FONT_HERSHEY_SIMPLEX,
                    ACTION_RECT_HEIGHT / 40, (255, 255, 255), 1, cv2.LINE_AA)
    return output_frame


def resize_image(image, width, height, inter=cv2.INTER_AREA):  # Переименована функция
    (h, w) = image.shape[:2]
    if width / w >= height / h:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)

    # Добавление границ
    vertical_border = (height - resized.shape[0]) // 2
    horizontal_border = (width - resized.shape[1]) // 2
    resized = cv2.copyMakeBorder(src=resized,
                                 top=vertical_border, bottom=vertical_border,
                                 left=horizontal_border, right=horizontal_border,
                                 borderType=cv2.BORDER_CONSTANT)
    return resized


sequence_data = []  # Переименована переменная
sentence_history = []  # Переименована переменная
predictions_list = []  # Переименована переменная
prediction_threshold = 0.5  # Переименована переменная
mp_holistic = mp.solutions.holistic
camera_index = 0  # Переименована переменная
cap = cv2.VideoCapture(camera_index)
WINDOW_TITLE = "OpenCV Feed"  # Переименована переменная
EXIT_KEY = 'qé'  # Переименована переменная
FULLSCREEN_KEY = 'fà'  # Переименована переменная
is_fullscreen_mode = True  # Переименована переменная

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        current_time = time.time()  # Переименована переменная
        if not cap.isOpened():
            image = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(image, f'NOT FOUND CAMERA {camera_index + 1}', (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cap = cv2.VideoCapture(camera_index)
        else:
            if is_fullscreen_mode:
                cv2.namedWindow(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                (x, y, windowWidth, windowHeight) = cv2.getWindowImageRect(WINDOW_TITLE)
            else:
                windowWidth = 640
                windowHeight = 480

            # Чтение видеопотока
            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue

            # Обработка изображений
            image, results = mediapipe_detection(frame, holistic)

            # Рисование ключевых точек
            draw_styled_landmarks_for_testing(image, results)
            if is_fullscreen_mode:
                image = resize_image(image, windowWidth, windowHeight, cv2.INTER_LINEAR)

            # Логика предсказания
            keypoints = extract_keypoints(results)[-126:]
            sequence_data.append(keypoints)
            sequence_data = sequence_data[-count_frames:]

            if len(sequence_data) == count_frames:
                res = model.predict(np.array([sequence_data]), verbose=0)[0]
                predictions_list.append(np.argmax(res))

                # Логика визуализации
                if np.unique(predictions_list[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > prediction_threshold:

                        if len(sentence_history) > 0:
                            if actions_list[np.argmax(res)] != sentence_history[-1]:
                                sentence_history.append(actions_list[np.argmax(res)])
                        else:
                            sentence_history.append(actions_list[np.argmax(res)])

                if len(sentence_history):
                 if len(sentence_history) > 10:  # Переименована переменная
                    sentence_history = sentence_history[-10:]  # Обновление списка предложений

                # Визуализация вероятностей
                image = visualize_probabilities(res, actions_list, image, colors)  # Переименована переменная

            cv2.rectangle(image, (0, 0), (windowWidth, 40), (9, 128, 246), -1)
            cv2.putText(image, ' '.join(sentence_history), (3, 30),  # Переименована переменная
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Отображение на экране
        cv2.imshow(WINDOW_TITLE, image)  # Переименована переменная
        if not is_fullscreen_mode:  # Переименована переменная
            # Всегда поверх других окон
            cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)  # Переименована переменная
        timeout = max(1, int(round(1000 * (1 / FPS - (time.time() - current_time)))))  # Переименована переменная

        # Корректный выход
        pressed_key = chr(cv2.waitKey(timeout) & 0xFF).lower()
        if pressed_key in EXIT_KEY:  # Переименована переменная
            break
        elif pressed_key.isdigit():
            camera_index = int(pressed_key) - 1  # Переименована переменная
            cap = cv2.VideoCapture(camera_index)  # Переименована переменная
    cap.release()
    cv2.destroyAllWindows()