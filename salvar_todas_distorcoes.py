import pandas as pd
from os import listdir, mkdir
from cv2 import imwrite
from classifier import classify_to_data, open_img, ssim
from funcoes import add_gaussian_noise, blur, alter_bright, canny, image_rotation


PATH_TO_DIR_VAL = "imagenette2/val/n01440764/"
PATH_TO_SAVE_VAL = "distorcoes/val/n01440764/"
PATH_TO_DIR_TRAIN = "imagenette2/train/n01440764/"
PATH_TO_SAVE_TRAIN = "distorcoes/train/n01440764/"


HEADERS = [
    "FILE_PATH",
    "ORIG_CLASSES",
    "GAUSS_SSIM",
    "GAUSS_CLASSES",
    "BLUR_SSIM",
    "BLUR_CLASSES",
    "GRAYSCALE_SSIM",
    "GRAYSCALE_CLASSES",
    "NEGATIVE_SSIM",
    "NEGATIVE_CLASSES",
    "ZOOM_SSIM",
    "ZOOM_CLASSES",
]
df = pd.DataFrame(columns=HEADERS)


try:
    mkdir("distorcoes")
except FileExistsError:
    pass

try:
    mkdir("distorcoes/val/")
except FileExistsError:
    pass

try:
    mkdir(PATH_TO_SAVE_VAL)
except FileExistsError:
    pass

try:
    mkdir("distorcoes/train/")
except FileExistsError:
    pass

try:
    mkdir(PATH_TO_SAVE_TRAIN)
except FileExistsError:
    pass


files_val = listdir(PATH_TO_DIR_VAL)
files_train = listdir(PATH_TO_DIR_TRAIN)


for file in files_val:
    path = PATH_TO_DIR_VAL + file

    save = PATH_TO_SAVE_VAL + file.removesuffix(".JPEG")
    try:
        mkdir(save)
    except FileExistsError:
        pass

    orig_img = open_img(path=path)

    classes_orig = classify_to_data(orig_img)

    #Canny

    after_canny = canny(orig_img)
    imwrite(f"{save}/{file}_dist_canny.JPEG", after_canny)

    ssim_canny = ssim(orig_img, after_canny)
    classes_canny = classify_to_data(after_canny)

    # Rotation

    after_rotation = image_rotation(orig_img)

    imwrite(f"{save}/{file}_dist_rotation.JPEG", after_rotation)

    ssim_rotation = ssim(orig_img, after_rotation)
    classes_rotation = classify_to_data(after_rotation)

    #Brightness

    after_brightness = alter_bright(orig_img,1.5,50)

    imwrite(f"{save}/{file}_dist_brightness.JPEG", after_brightness)

    ssim_brightness = ssim(orig_img, after_brightness)
    classes_brightness = classify_to_data(after_brightness)

    #Gaussian

    after_gaussian = add_gaussian_noise(orig_img,mean=0, sigma=25)

    imwrite(f"{save}/{file}_dist_gaussian.JPEG", after_gaussian)

    ssim_gaussian = ssim(orig_img, after_gaussian)
    classes_gaussian = classify_to_data(after_gaussian)

    #Blur

    after_blur = blur(orig_img)

    imwrite(f"{save}/{file}_dist_blur.JPEG", after_blur)

    ssim_blur = ssim(orig_img, after_blur)
    classes_blur = classify_to_data(after_blur)


    df.loc[-1] = [
        path.removeprefix("distorcoes/"),
        classes_orig,
        ssim_canny,
        classes_canny,
        ssim_rotation,
        classes_rotation,
        ssim_brightness,
        classes_brightness,
        ssim_gaussian,
        classes_gaussian,
        ssim_blur,
        classes_blur
    ]
    df.index = df.index + 1
    df = df.sort_index()

for file in files_train:
    path = PATH_TO_DIR_TRAIN + file

    save = PATH_TO_SAVE_TRAIN + file.removesuffix(".JPEG")
    try:
        mkdir(save)
    except FileExistsError:
        pass

    orig_img = open_img(path=path)

    classes_orig = classify_to_data(orig_img)

    #Canny

    after_canny = canny(orig_img)
    imwrite(f"{save}/{file}_dist_canny.JPEG", after_canny)

    ssim_canny = ssim(orig_img, after_canny)
    classes_canny = classify_to_data(after_canny)

    # Rotation

    after_rotation = image_rotation(orig_img)

    imwrite(f"{save}/{file}_dist_rotation.JPEG", after_rotation)

    ssim_rotation = ssim(orig_img, after_rotation)
    classes_rotation = classify_to_data(after_rotation)

    #Brightness

    after_brightness = alter_bright(orig_img,1.5,50)

    imwrite(f"{save}/{file}_dist_brightness.JPEG", after_brightness)

    ssim_brightness = ssim(orig_img, after_brightness)
    classes_brightness = classify_to_data(after_brightness)

    #Gaussian

    after_gaussian = add_gaussian_noise(orig_img,mean=0, sigma=25)

    imwrite(f"{save}/{file}_dist_gaussian.JPEG", after_gaussian)

    ssim_gaussian = ssim(orig_img, after_gaussian)
    classes_gaussian = classify_to_data(after_gaussian)

    #Blur

    after_blur = blur(orig_img)

    imwrite(f"{save}/{file}_dist_blur.JPEG", after_blur)

    ssim_blur = ssim(orig_img, after_blur)
    classes_blur = classify_to_data(after_blur)


    df.loc[-1] = [
        path.removeprefix("distorcoes/"),
        classes_orig,
        ssim_canny,
        classes_canny,
        ssim_rotation,
        classes_rotation,
        ssim_brightness,
        classes_brightness,
        ssim_gaussian,
        classes_gaussian,
        ssim_blur,
        classes_blur
    ]
    df.index = df.index + 1
    df = df.sort_index()

df.to_csv("distorcoes/distorcoes-data.csv", sep=";")