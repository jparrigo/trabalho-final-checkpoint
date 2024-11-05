#
# Depends on:
# - tensorflow: pip install tensorflow
# - keras: pip install keras
# - opencv: pip install opencv-python
# - scikit-image: pip install scikit-image
# Windows users may need to use "py -m pip install" or "python -m pip install" instead of "pip install".

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
import funcoes as fn
from skimage.metrics import structural_similarity

MODEL = ResNet50(weights='imagenet')
PATH = "imagenette2/val/n01440764/ILSVRC2012_val_00009111.JPEG"

def classify(img):
  try:
    x = cv2.resize(img, (224,224))
    x = x[:,:,::-1].astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = MODEL.predict(x)
    classes = decode_predictions(preds)[0]
    for c in classes:
      print("\t%s (%s): %.2f%%"%(c[1], c[0], c[2]*100))

  except Exception as e:
    print("Classification failed.")

def classify_to_data(img):
    classify = []
    try:
        x = cv2.resize(img, (224, 224))
        x = x[:, :, ::-1].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = MODEL.predict(x)
        classes = decode_predictions(preds)[0]
        for c in classes:
            classify.append([c[1], c[0], c[2] * 100])

    except Exception as e:
        print("Classification failed.")

    return classify

def open_img(path):
  return cv2.imread(path)

def ssim(img1, img2):
  return structural_similarity(img1, img2, channel_axis=2)*100

orig_img = open_img(PATH)

print("Original image:")
classify(orig_img)

# print("After Canny edge detection:")
# after_canny = fn.canny(orig_img)
# print("SSIM = %.2f"%(ssim(orig_img, after_canny)))
# classify(after_canny)
# cv2.imwrite("./distorcoes/after_canny.jpeg",after_canny)

# print("Depois de rotacionar imagem:")
# after_rotation = fn.image_rotation(orig_img)
# print("SSIM = %.2f"%(ssim(orig_img, after_rotation)))
# classify(after_rotation)
# cv2.imwrite("./distorcoes/after_rotation.jpeg",after_rotation)

# print("Depois de alterar brilho e contraste na imagem:")
# after_brightness = fn.alter_bright(orig_img,1.5,50)
# print("SSIM = %.2f"%(ssim(orig_img, after_brightness)))
# classify(after_brightness)
# cv2.imwrite("./distorcoes/after_brightness.jpeg",after_brightness)

# print("Depois de colocar o ruido gaussiano na imagem:")
# after_gaussian = fn.add_gaussian_noise(orig_img,mean=0, sigma=25)
# print("SSIM = %.2f"%(ssim(orig_img, after_gaussian)))
# classify(after_gaussian)
# cv2.imwrite("./distorcoes/after_gaussian.jpeg",after_gaussian)

# print("Depois de colocar o filtro de blur na imagem:")
# after_blur = fn.blur(orig_img)
# print("SSIM = %.2f"%(ssim(orig_img, after_blur)))
# classify(after_blur)
# cv2.imwrite("./distorcoes/after_blur.jpeg",after_blur)