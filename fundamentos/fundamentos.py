import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
import skimage.exposure

#lendo a imagen
i0 = cv2.imread('raioXTorax.pgm', 0)
i1 = cv2.imread('Lamina-biopsia.jpg',0)

#normalizando
in0 = skimage.img_as_float(i0)
in1 = skimage.img_as_float(i1)

(M,N) = np.shape(in0)
(m,n) = np.shape(in1)

#identificando o valor dos pixels
in0[50,50]
in1[50,50]

#max e min de cada imagem
maximo = np.max(in0)
minimo = np.min(in0)
media = np.mean(in0)
dP = np.std(in0)

maximo = np.max(in1)
minimo = np.min(in1)
media = np.mean(in1)
dP = np.std(in1)

#mostrando a imagem 0
plt.figure
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('image0')
plt.imshow(in0, cmap='gray') #--> Colormaps alternativo, verificar outros→
cmap='jet'
plt.colorbar()

#mostrando a imagem 1
plt.figure
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('image0')
plt.imshow(in1, cmap='gray') #--> Colormaps alternativo, verificar outros→
cmap='jet'
plt.colorbar()



#lendo a imagem
i1 = cv2.imread('Lamina-biopsia.jpg',1)

#mostrando a imagem 1 e tons de azul
plt.figure
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('blue')
plt.imshow(in1[:,:,0], cmap='Blues') #--> Colormaps alternativo, verificar outros→
plt.colorbar()

#mostrando a imagem 1 em tons de verde
plt.figure
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('green')
plt.imshow(in1[:,:,2], cmap='Greens') #--> Colormaps alternativo, verificar outros→
plt.colorbar()

#mostrand a imagem 1 em tons de vermelho
plt.figure
plt.ylabel('linhas - M')
plt.xlabel('colunas - N')
plt.title('red')
plt.imshow(in1[:,:,2], cmap='Reds') #--> Colormaps alternativo, verificar outros→
plt.colorbar()