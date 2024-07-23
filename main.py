import cv2
from cvzone.FaceDetectionModule import FaceDetector
from deepface import DeepFace
import numpy as np
import tensorflow as tf

# Configurar TensorFlow para usar GPU, se disponível
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limitar a memória da GPU, se necessário
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Ajuste conforme necessário
    except RuntimeError as e:
        print(e)

# Inicializar a captura de vídeo e o detector de rostos
video = cv2.VideoCapture(0)
detector = FaceDetector()
running = True  # Variável para controlar o loop

# Função para reconhecer emoções
def recognize_emotion(image):
    try:
        # Usar DeepFace para reconhecer a emoção na imagem
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"Erro ao reconhecer emoções: {e}")
        return None

while running:
    _, img = video.read()
    original_img = img.copy()  # Fazer uma cópia da imagem original
    img, bboxes = detector.findFaces(img, draw=True)

    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            face_img = original_img[y:y+h, x:x+w]

            # Reconhecer a emoção da imagem do rosto
            emotion = recognize_emotion(face_img)
            if emotion:
                cv2.putText(img, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                print(f"Emotion detected: {emotion}")

    cv2.imshow('Resultado', img)
    key = cv2.waitKey(1)
    if key == 27:  # Pressione 'ESC' para sair
        running = False

video.release()
cv2.destroyAllWindows()
