#AGCC - Adaptive Gamma and Contrast Correction

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2

# read and open file
fileName = input('file name: ')
img = Image.open(fileName)

# calculate image properties
def cip(img):
    # turn to NumPy array
    image = np.array(img)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_mean = np.mean(rgb[:, :, 0])
    g_mean = np.mean(rgb[:, :, 1])
    b_mean = np.mean(rgb[:, :, 2])
    
    brightness = 0.2126*r_mean + 0.7152*g_mean + 0.0722*b_mean
    C_avg = r_mean + g_mean + b_mean
    contrast = gray.std()
    saturation = hsv[:, :, 1].mean()
    return brightness, contrast, saturation, C_avg

def gamma_enhance(img, A, C_avg, L):
    img_f = img.astype(np.float32)
    
    # avoid to div by 0
    if C_avg == 0:
        gamma = 1.0
    else:
        gamma = A * (L / C_avg)   # adaptive gamma
    
    print(f"gamma: {gamma:.3f}")
    # normalize to [0,1]
    img_norm = img_f / 255.0
    
    # correct gamma
    img_corrected = np.power(img_norm, 2.0 / gamma)
    
    # to [0,255]
    img_out = np.clip(img_corrected * 255.0, 0, 255).astype(np.uint8)
    
    return img_out, gamma

brightness ,contrast, saturation, C_avg = cip(img)
# auto calculate gamma
A = 500 / contrast
A = min(A, 25)
print(A)
#turn to numpy array
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# denoise
brightness ,contrast, saturation, C_avg = cip(img)
denoise_rate = 100 / brightness
img = cv2.fastNlMeansDenoisingColored(img, None, denoise_rate, denoise_rate, 7, 21)
# correct gamma
img, gamma = gamma_enhance(img, A, C_avg, brightness)
# turn to pillow image
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(rgb)

# enhance contrast
brightness ,contrast, saturation, C_avg = cip(img)
contrast_enhancer = ImageEnhance.Contrast(img)
img = contrast_enhancer.enhance((50/contrast+gamma / 5)/1.5)
print((50/contrast+gamma / 5)/1.5)

# save
img.save('output.png')
