import pandas as pd
import os
from cv2 import imwrite
import numpy as np
from classifier import classify_to_data, open_img, ssim
from funcoes import add_gaussian_noise, blur, alter_bright, canny, image_rotation

# Diretórios de entrada e saída
PATH_TO_DIR_VAL = "imagenette2/val/n01440764/"
PATH_TO_SAVE_VAL = "distorcoes/val/n01440764/"
PATH_TO_DIR_TRAIN = "imagenette2/train/n01440764/"
PATH_TO_SAVE_TRAIN = "distorcoes/train/n01440764/"

# Ajustando o cabeçalho para refletir as distorções aplicadas e métricas adicionais
HEADERS = [
    "FILE_PATH",
    "ORIG_CLASSES",
    "CANNY_SSIM",
    "CANNY_CLASSES",
    "CANNY_EDGE_RATIO",
    "CANNY_EDGE_INTENSITY",
    "ROTATION_SSIM",
    "ROTATION_CLASSES",
    "BRIGHTNESS_SSIM",
    "BRIGHTNESS_CLASSES",
    "GAUSSIAN_SSIM",
    "GAUSSIAN_CLASSES",
    "BLUR_SSIM",
    "BLUR_CLASSES",
]
df = pd.DataFrame(columns=HEADERS)

# Função para calcular a proporção de bordas
def calculate_edge_ratio(edge_image):
    edge_pixels = np.sum(edge_image == 255)
    total_pixels = edge_image.size
    return edge_pixels / total_pixels if total_pixels > 0 else 0

# Função para calcular a intensidade média das bordas
def calculate_edge_intensity(edge_image):
    return np.mean(edge_image[edge_image == 255]) if np.any(edge_image == 255) else 0

# Criar diretórios de saída, se não existirem
for path in ["distorcoes", "distorcoes/val", PATH_TO_SAVE_VAL, "distorcoes/train", PATH_TO_SAVE_TRAIN]:
    os.mkdir(path) if not os.path.exists(path) else None

# Processar as imagens para validação e treinamento
for files, path_to_dir, path_to_save in [(os.listdir(PATH_TO_DIR_VAL), PATH_TO_DIR_VAL, PATH_TO_SAVE_VAL),
                                         (os.listdir(PATH_TO_DIR_TRAIN), PATH_TO_DIR_TRAIN, PATH_TO_SAVE_TRAIN)]:

    for file in files:
        path = path_to_dir + file
        save = path_to_save + file.removesuffix(".JPEG")
        os.mkdir(save) if not os.path.exists(save) else None

        orig_img = open_img(path=path)
        classes_orig = classify_to_data(orig_img)

        # Canny
        after_canny = canny(orig_img)
        imwrite(f"{save}/{file}_dist_canny.JPEG", after_canny)
        ssim_canny = ssim(orig_img, after_canny)
        classes_canny = classify_to_data(after_canny)
        edge_ratio_canny = calculate_edge_ratio(after_canny)
        edge_intensity_canny = calculate_edge_intensity(after_canny)

        # Rotation
        after_rotation = image_rotation(orig_img)
        imwrite(f"{save}/{file}_dist_rotation.JPEG", after_rotation)
        ssim_rotation = ssim(orig_img, after_rotation)
        classes_rotation = classify_to_data(after_rotation)

        # Brightness
        after_brightness = alter_bright(orig_img, 1.5, 50)
        imwrite(f"{save}/{file}_dist_brightness.JPEG", after_brightness)
        ssim_brightness = ssim(orig_img, after_brightness)
        classes_brightness = classify_to_data(after_brightness)

        # Gaussian
        after_gaussian = add_gaussian_noise(orig_img, mean=0, sigma=25)
        imwrite(f"{save}/{file}_dist_gaussian.JPEG", after_gaussian)
        ssim_gaussian = ssim(orig_img, after_gaussian)
        classes_gaussian = classify_to_data(after_gaussian)

        # Blur
        after_blur = blur(orig_img)
        imwrite(f"{save}/{file}_dist_blur.JPEG", after_blur)
        ssim_blur = ssim(orig_img, after_blur)
        classes_blur = classify_to_data(after_blur)

        # Adicionando a linha de dados ao DataFrame com as novas métricas de Canny
        df.loc[-1] = [
            path.removeprefix("distorcoes/"),
            classes_orig,
            ssim_canny,
            classes_canny,
            edge_ratio_canny,
            edge_intensity_canny,
            ssim_rotation,
            classes_rotation,
            ssim_brightness,
            classes_brightness,
            ssim_gaussian,
            classes_gaussian,
            ssim_blur,
            classes_blur,
        ]
        df.index = df.index + 1
        df = df.sort_index()

# Salvando o DataFrame atualizado em um arquivo CSV
df.to_csv("distorcoes/distorcoes-data.csv", sep=";")
