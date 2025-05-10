"""
Este modulo permite detectar el color dominante y el numero central en una carta de UNO.
Usa procesamiento de imagen con OpenCV y OCR con EasyOCR.
"""

import cv2 as cv
import numpy as np
import easyocr
import warnings

# Ignora advertencias de rendimiento de PyTorch relacionadas con memoria
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*",
    category=UserWarning,
    module=r"torch\.utils\.data\.dataloader"
)


def mostrar_ventana_red(titulo, img, scale=0.25):
    """
    Muestra una ventana con la imagen redimensionada para facilitar la visualizacion.
    """
    h, w = img.shape[:2]
    img_peq = cv.resize(img, (int(w * scale), int(h * scale)))
    cv.imshow(titulo, img_peq)


def detect_uno_color(img_bgr, mostrar_pasos=True):
    """
    Detecta el color dominante de una carta de UNO.
    Utiliza el espacio de color HSV y un histograma de tonos.

    Parametros:
        img_bgr (np.ndarray): Imagen de entrada en formato BGR.
        mostrar_pasos (bool): Si es True, muestra visualizacion paso a paso.

    Retorna:
        str: Color detectado ('red', 'yellow', 'green', 'blue').
    """
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = (s > 100) & (v > 100)
    mask_visual = mask.astype(np.uint8) * 255
    h_masked = h[mask]

    if mostrar_pasos:
        mostrar_ventana_red("1 - Imagen original", img_bgr)
        mostrar_ventana_red("2 - HSV completo", hsv)
        mostrar_ventana_red("3 - Mascara S>100 y V>100", mask_visual)

    hist = cv.calcHist([h_masked], [0], None, [180], [0, 180]).flatten()
    dom = np.argmax(hist)

    if mostrar_pasos:
        cv.waitKey(0)
        cv.destroyAllWindows()

    if dom < 10 or dom >= 170:
        return 'red'
    elif dom < 30:
        return 'yellow'
    elif dom < 85:
        return 'green'
    else:
        return 'blue'


def read_uno_number(img_bgr, mostrar_pasos=True):
    """
    Extrae el numero central de una carta UNO utilizando reconocimiento optico (OCR).

    Parametros:
        img_bgr (np.ndarray): Imagen en color de la carta.
        mostrar_pasos (bool): Si es True, muestra visualmente los pasos aplicados.

    Retorna:
        str o None: Numero detectado como cadena o None si no se detecta nada.
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    if mostrar_pasos:
        mostrar_ventana_red("1 - Gris", gray)

    _, bw = cv.threshold(gray, 155, 255, cv.THRESH_BINARY_INV)
    if mostrar_pasos:
        mostrar_ventana_red("2 - Mascara", bw)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    limpia = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    if mostrar_pasos:
        mostrar_ventana_red("3 - Limpia", limpia)

    difum = cv.GaussianBlur(limpia, (127, 127), 0)
    if mostrar_pasos:
        mostrar_ventana_red("4 - Difusa", difum)

    _, final = cv.threshold(difum, 220, 255, cv.THRESH_BINARY)
    if mostrar_pasos:
        mostrar_ventana_red("5 - Final", final)

    if mostrar_pasos:
        cv.waitKey(0)
        cv.destroyAllWindows()

    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    textos = reader.readtext(final, detail=0)
    digitos = [t for t in textos if t.strip().isdigit()]
    return max(digitos, key=len) if digitos else None
