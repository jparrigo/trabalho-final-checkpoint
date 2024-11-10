import cv2
import numpy as np

# Função de canny
def canny(img):
  x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  x = cv2.Canny(x, 100, 200)
  return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

# Função para rotacionar 45 graus
def image_rotation(img):
  (alt, lar) = img.shape[:2]
  centro = (lar // 2, alt // 2)
  M = cv2.getRotationMatrix2D(centro, 45, 1.0)
  img_rot = cv2.warpAffine(img, M, (lar, alt))

  return img_rot

# Alterar brilho e contraste
def alter_bright(img,alpha,beta):
  return cv2.convertScaleAbs(img,alpha=alpha,beta=beta)

#Função ruido gaussiano
def add_gaussian_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy_img = cv2.add(img.astype(np.float32), gaussian_noise)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img

# Função de blur
def blur(img):
  return cv2.blur(img,ksize=(3,3))