import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import scipy.signal

#criando um vetor amostra
amostra = [15, 29, 5, 8, 255, 40, 1, 0, 10]

#ordenando o vetor amostra
amostraOrdenada = np.sort(amostra)

#mediana da amostra ordenada
mediana = np.median(amostraOrdenada)

#lendo e normalizando a imagem
lMRI = cv2.imread('TransversalMRI_salt-and-pepper.pgm',0)
lMRI_N = skimage.img_as_float(lMRI)

#exibindo as imagens
plt.figure()
plt.title('IMRI Normalizada')
plt.imshow(lMRI_N, cmap='gray') # cmap='jet'

#criando uma matriz de zeros
(M,N) = np.shape(lMRI)
lMRIfiltrada = np.zeros((M,N), float)

#criando a máscara 3x3
w = np.ones((3,3))

#varrendo a imagem com a máscara 3x3
for l in range(M-2):
    for c in range(N-2):
        matAmostras = (w*lMRI[(l):(l)+3,(c):(c)+3])
        vetorAmostras = np.concatenate((matAmostras), axis = None)
        vetorAmostrasOrdenadas = np.sort(vetorAmostras)
        lMRIfiltrada[l+1,c+1] = vetorAmostrasOrdenadas[4]

plt.figure()
plt.title('IMRI Filtrada')
plt.imshow(lMRIfiltrada, cmap='gray') # cmap='jet'

#testando a rotina usando a função
teste = scipy.signal.medfilt2d(lMRI)
plt.figure()
plt.title('IMRI Filtrada usando uma função')
plt.imshow(teste, cmap='gray') # cmap='jet'

#usando a função gaussiana
g = scipy.signal.gaussian(9,std=1)

g1 = np.zeros((9,9), float)
g1[4,:] = g
gtranspose1 = np.transpose(g1)
w_Gauss2D = scipy.signal.convolve2d(g1,gtranspose1,'same')
w_Gauss2DNormal = w_Gauss2D/(np.sum(w_Gauss2D))

#plotando a máscara
plt.figure()
plt.title('w_Gauss2D ')
plt.imshow(w_Gauss2DNormal , cmap='gray') # cmap='jet'

#convoluindo a imagem mamography e a máscara
Mamo = cv2.imread('Mamography.pgm',0)
MamoN = skimage.img_as_float(Mamo)

MamoFilt = scipy.signal.convolve2d(Mamo,w_Gauss2DNormal,'same')

plt.figure()
plt.title('MamoFilt')
plt.imshow(MamoFilt , cmap='gray') # cmap='jet'

#chamando a função
import bibMascara

#chamando a função com as entradas e saídas
w_Gauss2DNormalizado = bibMascara.fazerMascaraGauss2D(media=7, desvio=3)

plt.figure()
plt.title('w_Gauss2DNormalizado media = 7 e desvio = 3')
plt.imshow(w_Gauss2DNormalizado , cmap='gray') # cmap='jet'

#testando para diferentes valores
w_Gauss2DNormalizado = bibMascara.fazerMascaraGauss2D(media=4, desvio=1)

plt.figure()
plt.title('w_Gauss2DNormalizado media = 4 e desvio = 1')
plt.imshow(w_Gauss2DNormalizado , cmap='gray') # cmap='jet'

w_Gauss2DNormalizado = bibMascara.fazerMascaraGauss2D(media=7, desvio=5)

plt.figure()
plt.title('w_Gauss2DNormalizado media = 7 e desvio = 5')
plt.imshow(w_Gauss2DNormalizado , cmap='gray') # cmap='jet'