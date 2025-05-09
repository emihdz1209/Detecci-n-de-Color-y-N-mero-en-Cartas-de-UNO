# Detección de Color y Número en Cartas de UNO

Este proyecto en Python utiliza procesamiento de imágenes (OpenCV) y reconocimiento óptico de caracteres (EasyOCR) para identificar el **color predominante** y el **número central** en cartas del juego UNO a partir de imágenes.

## Funcionalidad

El script:

1. Procesa una carpeta de imágenes `Cartas_en_orden` (formato `Card_*.jpg`).
2. Extrae el color dominante de cada carta usando el espacio de color HSV.
3. Detecta el número central mediante un pipeline de procesamiento de imagen y OCR.
4. Verifica que cada carta pueda colocarse legalmente encima de la anterior (coincide color o número).
5. Muestra el resultado del análisis con imágenes interactivas y un resumen textual.

## Estructura del Proyecto

- `UNO.py`: Script principal que coordina el procesamiento de imágenes y verifica la validez de las jugadas.
- `color_numero.py`: Contiene funciones auxiliares para:
  - Detección del color de la carta (`detect_uno_color`)
  - Reconocimiento del número con OCR (`read_uno_number`)
- `Cartas_en_orden/`: Carpeta esperada con imágenes de cartas nombradas como `Card_1.jpg`, `Card_2.jpg`, etc.
- `SUCCESS.jpg` y `FAIL.jpg`: Imágenes mostradas al final como resultado visual del análisis.

## Requisitos

- Python 3.x
- [OpenCV](https://pypi.org/project/opencv-python/)
- [EasyOCR](https://pypi.org/project/easyocr/)
- GPU compatible con CUDA (recomendado para EasyOCR, pero opcional)

### Instalación de dependencias

```bash
pip install opencv-python easyocr numpy
```

## Uso

1. Coloca tus imágenes en la carpeta `Cartas_en_orden/`, titulando cada una como "Card_Num.jpg"
2. Ejecuta el script principal:

```bash
python UNO.py
```

## Ejemplo de salida

```
--- Procesando Card_1.jpg
Color: red | Número: 5
--- Procesando Card_2.jpg
Color: red | Número: 7
✅ Todas encajan. ¡Bien jugado!
```
