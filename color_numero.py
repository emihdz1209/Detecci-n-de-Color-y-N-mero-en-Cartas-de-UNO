
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
def detect_uno_color(img_bgr, mostrar_pasos=True):
    # Pasa a HSV y separa canales sin aplanar
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Filtra píxeles bien saturados y brillantes
    mask = (s > 100) & (v > 100)
    mask_visual = mask.astype(np.uint8) * 255  # Para mostrar

    # Aplicar máscara a h
    h_masked = h[mask]

    # Mostrar pasos si se solicita
    if mostrar_pasos:
        mostrar_ventana_red("1 - Imagen original", img_bgr)
        mostrar_ventana_red("2 - HSV completo", hsv)
        mostrar_ventana_red("3 - Mascara S>100 y V>100", mask_visual)

    # Histograma de tonos
    hist = cv.calcHist([h_masked], [0], None, [180], [0, 180]).flatten()
    dom = np.argmax(hist)

    # Pausar si se muestran pasos
    if mostrar_pasos:
        cv.waitKey(0)
        cv.destroyAllWindows()

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
    return max(digitos, key=len) if digitos else None