#importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import scipy.signal
from math import pi

#criando o vetor tempo
t = np.arange(0,10,0.01)

#variáveis das frequências
f1 = 1 #1Hz
f2 = 3 #3Hz
f3 = 5 #5Hz

#criando as funções senoidais
s1 = np.sin(2*pi*t*f1)
s2 = np.sin(2*pi*t*f2)
s3 = np.sin(2*pi*t*f3)

sinal = s1+s2+s3

#plot dos sinais
plt.figure(1)
plt.subplot(4,1,1) #l,c,pos
plt.plot(t,s1)
plt.ylabel('S1')

plt.subplot(4,1,2)
plt.plot(t,s2)
plt.ylabel('S2')

plt.subplot(4,1,3)
plt.plot(t,s3)
plt.ylabel('S3')

plt.subplot(4,1,4)
plt.plot(t,sinal)
plt.ylabel('Sinal')

#Transformada de Fourier
Modulo = np.zeros(11)
for f in range(0,11):
    euler = np.exp(-1j*2*(np.math.pi)*f*t)
    X = np.sum(sinal*euler)
    Modulo[f] = np.abs(X)
    
plt.figure(2)
plt.stem(Modulo)

#Lendo o pulso quadrado
pulsoQuadrado = cv2.imread('PulsoQuadrado1.pgm',0)
pulsoQuadrado = skimage.img_as_float(pulsoQuadrado) #normalizando a imagem

#exibindo o pulso quadrado
plt.figure()
plt.title('Pulso Quadrado')
plt.plot(pulsoQuadrado)

#Transformada de Fourier 2D da imagem
PulsoFrequencia = np.fft.fft(pulsoQuadrado)
plt.figure()
plt.title('ModuloPulsoFrequencia')
plt.imshow(np.abs(PulsoFrequencia),cmap='gray')

FrequenciaDeslocado = np.fft.fftshift(PulsoFrequencia)
plt.figure()
plt.title('FrequenciaDeslocado')
plt.imshow(np.abs(FrequenciaDeslocado),cmap='gray')

#visualizando a figura com o log da frequencia deslocada
plt.figure()
plt.title('FrequenciaDeslocado em logaritmo')
plt.imshow(np.log(1+np.abs(FrequenciaDeslocado)),cmap='gray')

#criando um filtro passa baixas ideal que passe até 10% da intensidade máxima
(M,N) = np.shape(FrequenciaDeslocado)
H = np.zeros((M,N),complex)
fc = 0.1
Do = fc*(M/2)

for l in range(M):
    for c in range(N):
        distc = c-(N/2)
        distl = l-(M/2)
        dist = np.math.sqrt(pow(distc,2)+pow(distl,2))
        if dist < Do:
            H[l,c] = 1+0j

Ffiltrado = FrequenciaDeslocado*H
plt.figure()
plt.title('Ffiltrado')
plt.imshow(np.abs(Ffiltrado),cmap='gray')

#Transformada de Fourier Inversa de Ffiltrado
ifftFfiltrado = np.fft.ifft2(Ffiltrado)

plt.figure()
plt.title('ifftFfiltrado')
plt.imshow(np.abs(ifftFfiltrado),cmap='gray')

#lendo e normalizando a imagem de mamografia
Mamo = cv2.imread('Mamography.pgm',0)
MamoN = skimage.img_as_float(Mamo)

plt.figure()
plt.title('Mamography')
plt.imshow(MamoN,cmap='gray')

#Transformada da imagem de mamografia
fftMamo =np.fft.fftshift(np.fft.fft2(MamoN))
plt.figure()
plt.title('FT Mamo')
plt.imshow(np.abs(fftMamo),cmap='gray')

#Filtro H para a Mamografia
(M,N) = np.shape(fftMamo)
H = np.zeros((M,N),complex)
fc = 0.1
Do = fc*(M/2)

for l in range(M):
    for c in range(N):
        distc = c-(N/2)
        distl = l-(M/2)
        dist = np.math.sqrt(pow(distc,2)+pow(distl,2))
        if dist < Do:
            H[l,c] = 1+0j

MamoFiltrado = fftMamo*H
plt.figure()
plt.title('MamoFiltrado')
plt.imshow(np.abs(MamoFiltrado),cmap='gray')

#Transformada Inversa
ifftMamo = np.fft.ifft2(MamoFiltrado)

plt.figure()
plt.title('TF Inversa de Mamo')
plt.imshow(np.abs(ifftMamo),cmap='gray')