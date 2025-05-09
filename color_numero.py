import cv2 as cv
import numpy as np
import easyocr
import warnings

# Ignora warnings molestos de pin_memory
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*",
    category=UserWarning,
    module=r"torch\.utils\.data\.dataloader"
)

def mostrar_ventana_red(titulo, img, scale=0.25):
    # Redimensiona y muestra la imagen
    h, w = img.shape[:2]
    img_peq = cv.resize(img, (int(w*scale), int(h*scale)))
    cv.imshow(titulo, img_peq)

# ------------- Detección de color UNO -------------
def detect_uno_color(img_bgr):
    # Pasa a HSV y separa canales
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h = hsv[:, :, 0].flatten()
    s = hsv[:, :, 1].flatten()
    v = hsv[:, :, 2].flatten()
    # Filtra píxeles bien saturados y brillantes
    mask = (s > 100) & (v > 100)
    h = h[mask]
    # Histograma de tonos
    hist = cv.calcHist([h], [0], None, [180], [0, 180]).flatten()
    dom = np.argmax(hist)
    # Mapea al color UNO
    if dom < 10 or dom >= 170:
        return 'red'
    elif dom < 30:
        return 'yellow'
    elif dom < 85:
        return 'green'
    else:
        return 'blue'

# ------------- OCR para número central -------------
def read_uno_number(img_bgr, mostrar_pasos=True):
    # 1) Escala a grises
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    if mostrar_pasos:
        mostrar_ventana_red("1 - Gris", gray)

    # 2) Umbral inverso para resaltar trazos oscuros
    _, bw = cv.threshold(gray, 155, 255, cv.THRESH_BINARY_INV)
    if mostrar_pasos:
        mostrar_ventana_red("2 - Mascara", bw)

    # 3) Apertura morfológica para quitar ruiditos
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    limpia = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    if mostrar_pasos:
        mostrar_ventana_red("3 - Limpia", limpia)

    # 4) Blur gaussiano para difuminar detalles pequeños
    difum = cv.GaussianBlur(limpia, (127, 127), 0)
    if mostrar_pasos:
        mostrar_ventana_red("4 - Difusa", difum)

    # 5) Re-umbraliza tras difuminar
    _, final = cv.threshold(difum, 220, 255, cv.THRESH_BINARY)
    if mostrar_pasos:
        mostrar_ventana_red("5 - Final", final)

    # Pausa para ver los pasos
    if mostrar_pasos:
        cv.waitKey(0)
        cv.destroyAllWindows()

    # 6) Ejecuta OCR sobre la máscara final
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    textos = reader.readtext(final, detail=0)
    digitos = [t for t in textos if t.strip().isdigit()]
    # Devuelve el dígito más grande (el central)
    return max(digitos, key=len) if digitos else None