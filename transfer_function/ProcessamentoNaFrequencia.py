import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import scipy.signal

import bibFuncaoTransferencia

H_Ideal = bibFuncaoTransferencia.fazerMascaraIdeal(400,400,0.2)

#plot dos sinais
plt.figure()
plt.title('Mascara de Filtro Passa Baixa Ideal')
plt.imshow(np.abs(H_Ideal),cmap='gray')

H_Gauss = bibFuncaoTransferencia.fazerMascaraGaussiana2D(400,400,0.1)

#plot dos sinais
plt.figure()
plt.title('Mascara de Filtro Gaussiana')
plt.imshow(np.abs(H_Gauss),cmap='gray')

H_Butter = bibFuncaoTransferencia.fazerMascaraButterworth(400,400,0.1,2)

#plot dos sinais
plt.figure()
plt.title('Mascara de Filtro Butterworth')
plt.imshow(np.abs(H_Butter),cmap='gray')

#lendo a imagem mamografia
Mamo = cv2.imread('Mamography.pgm',0)

#normalizando a imagem
MamoN = skimage.img_as_float(Mamo)

#resize da imagem
MamoResize = cv2.resize(MamoN, (400,400))

#passando a imagem lida no domínio da frequência
ImFrequencia = np.fft.fft2(MamoResize)
ImFrequencia = np.fft.fftshift(ImFrequencia)

F_filtrada = ImFrequencia*H_Ideal
plt.figure()
plt.title('F_filtrada')
plt.imshow(np.abs(F_filtrada),cmap='gray')

#TF inversa
ifftF_filtrada = np.fft.ifft2(F_filtrada)
plt.figure()
plt.title('TF Inversa')
plt.imshow(np.abs(ifftF_filtrada),cmap='gray')


