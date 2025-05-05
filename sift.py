import cv2
import matplotlib.pyplot as plt

# Cargar la imagen
imagen = cv2.imread('chihuahua.jpg')

# Convertir la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Inicializar el detector SIFT
sift = cv2.SIFT_create()

# Detectar puntos clave y descriptores con SIFT
puntos_clave, descriptores = sift.detectAndCompute(imagen_gris, None)

# Dibujar los puntos clave en la imagen
imagen_con_puntos = cv2.drawKeypoints(
    imagen, puntos_clave, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen con los puntos clave
plt.imshow(cv2.cvtColor(imagen_con_puntos, cv2.COLOR_BGR2RGB))
plt.title('Puntos clave detectados con SIFT')
plt.show()
