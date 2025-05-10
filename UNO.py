"""
Script principal para procesar cartas UNO en imagen.
Verifica si una secuencia de cartas es valida visualmente segun color o numero.
"""

import os
import cv2 as cv
import color_numero


def show_image_rescale(img_title, img, scale=0.25):
    """
    Muestra una imagen redimensionada en una ventana con el titulo indicado.
    """
    h, w = img.shape[:2]
    img_peq = cv.resize(img, (int(w * scale), int(h * scale)))
    cv.imshow(img_title, img_peq)


# Ruta de trabajo actual
dir_actual = os.getcwd()

# Lista de cartas en orden alfabetico (Card_*.jpg)
cartas = sorted([
    f for f in os.listdir(os.path.join(dir_actual, "Cartas_en_orden"))
    if f.startswith("Card_") and f.endswith(".jpg")
])

prev_color = None
prev_num = None
fail = False

# Procesamiento de cada carta
for nombre in cartas:
    print(f"--- Procesando {nombre}")
    ruta = os.path.join(dir_actual, "Cartas_en_orden", nombre)

    img = cv.imread(ruta)
    if img is None:
        print(f"No pude abrir {nombre} --> salto")
        continue

    color = color_numero.detect_uno_color(img, False)  # Cambiar a True para ver pasos
    num = color_numero.read_uno_number(img, False)

    show_image_rescale(f"{color} {num}", img)

    print(f"Color: {color} | Numero: {num}")
    cv.waitKey(0)

    if prev_color is not None and prev_num is not None:
        if color != prev_color and num != prev_num:
            print("Ni color ni numero coinciden con la anterior")
            print(f"Antes: {prev_color} {prev_num} | Ahora: {color} {num}")
            fail = True
            break

    prev_color, prev_num = color, num

# Resultado final
if fail:
    show_image_rescale("Juego ilegal", cv.imread("FAIL.jpg"))
    cv.waitKey(0)
else:
    print("✅ Todas encajan. Bien jugado!")
    show_image_rescale("Juego valido", cv.imread("SUCCESS.jpg"))
    cv.waitKey(0)

cv.destroyAllWindows()
