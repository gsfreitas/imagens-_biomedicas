#importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import scipy.signal
from math import pi

#lendo a imagem
Ultrassom = cv2.imread('UltrassomBebe.pgm',0)

#normalizando a imagem
UltrassomN = skimage.img_as_float(Ultrassom)

#plot do ultrassom
plt.figure()
plt.title('Ultrassom')
plt.imshow(UltrassomN,cmap='gray')

#aplicando a função ROI
UltrassomROI = cv2.selectROI(UltrassomN)

#obtendo a média e a variância
Cmin = UltrassomROI[0]
Lmin = UltrassomROI[1] 
Cmax = UltrassomROI[0]+UltrassomROI[2]
Lmax = UltrassomROI[1]+UltrassomROI[3]

media = np.mean(Ultrassom[Lmin:Lmax,Cmin:Cmax])
varRegHomo = np.var(Ultrassom[Lmin:Lmax,Cmin:Cmax])

#definindo uma máscara 7x7
w = np.ones([7,7])/49

#definindo o tamanho da imagem ultrassom normalizada
(M,N) = np.shape(UltrassomN)

Img = UltrassomN
ImgMedia = scipy.signal.convolve2d(UltrassomN,w,'same')
ImgLee = np.zeros((M,N))
k = np.zeros((M,N))

for l in range(M-7):
    for c in range(N-7):
        varLocal = np.var(UltrassomN[l:l+7, c:c+7]) + 0.000001
        k[l+3,c+3] = 1 - (varRegHomo/varLocal)

k = np.clip(k,0,1)

ImgLee = ImgMedia + k*(Img - ImgMedia)

#plot do filtro de Lee
plt.figure()
plt.title('Ultrassom Filtro de Lee')
plt.imshow(ImgLee,cmap='gray')

#criando uma mascara/filtro de sobel
wxSobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
wySobel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

dfdx = scipy.signal.convolve2d(UltrassomN,wxSobel,'same')
dfdy = scipy.signal.convolve2d(UltrassomN,wySobel,'same')

moduloSobel = np.square(pow(dfdx,2)+pow(dfdy,2))

k = np.clip(moduloSobel,0,1)

ImgLee = ImgMedia + k*(Img - ImgMedia)

#plot com o filtro sobel
plt.figure()
plt.title('Ultrassom Filtro de Lee usando Sobel')
plt.imshow(ImgLee,cmap='gray')