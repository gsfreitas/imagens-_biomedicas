#importando as bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import skimage.exposure
import scipy.signal
import math

#lendo as imagens
ImSemRuido = cv2.imread('ImSemRuido.pgm', 0)

#normalizando a imagem
ImSemRuido = skimage.img_as_float(ImSemRuido)

#plot da imagem
plt.figure()
plt.title('Imagem sem Ruido')
plt.imshow(ImSemRuido,cmap='gray')

#1 PASSO
#colocando ruidos na imagem
ImComRuido = skimage.util.random_noise(ImSemRuido, mode='gaussian', seed=None, clip=True, mean=0, var=0.05)

#dimensoes da imagem
(M,N) = np.shape(ImComRuido)

#plot da imagem
plt.figure()
plt.title('Imagem com Ruido')
plt.imshow(ImComRuido,'gray')

(M,N) = np.shape(ImComRuido)

Eqmn = np.math.sqrt(pow(np.sum(ImSemRuido-ImComRuido),2)/(N*M))
Emax = np.max(np.abs(ImSemRuido-ImComRuido))

CoVar = np.sum((ImSemRuido - np.mean(ImSemRuido)) * (ImComRuido - np.mean(ImComRuido)))/(M*N)

p1 = CoVar/(np.std(ImSemRuido)*np.std(ImComRuido))
p2 = (2*(np.mean(ImComRuido) * np.mean(ImSemRuido))) / (math.pow(np.mean(ImComRuido),2) + math.pow(np.mean(ImSemRuido),2))
p3 = (2*(np.std(ImComRuido) * np.std(ImSemRuido))) / (math.pow((np.std(ImComRuido)),2) + math.pow((np.std(ImSemRuido)),2))

Q = p1 * p2 *p3

#2

#a - filtro tipo média simples

dim= 7
ponto_medio = int(np.floor(dim/2)) #pto central da máscara

w_media = np.ones((dim,dim),float)/(dim*dim)
ImFilt_media = scipy.signal.convolve2d(ImComRuido,w_media,'same')

plt.figure()
plt.title('Filtro tipo média semples')
plt.imshow(ImFilt_media,cmap='gray')

Eqmn_media = math.pow(np.sum(ImSemRuido-ImFilt_media),2)/(N*M)*0.5
Emax_media = np.max(np.abs(ImSemRuido-ImFilt_media))

CoVar_media = np.sum((ImSemRuido - np.mean(ImSemRuido)) * (ImFilt_media - np.mean(ImFilt_media)))/(M*N)

p1_media = CoVar_media/(np.std(ImSemRuido)*np.std(ImFilt_media))
p2_media = (2*(np.mean(ImFilt_media) * np.mean(ImSemRuido))) / (math.pow(np.mean(ImFilt_media),2) + math.pow(np.mean(ImSemRuido),2))
p3_media = (2*(np.std(ImFilt_media) * np.std(ImSemRuido))) / (math.pow(np.std(ImFilt_media),2) + math.pow(np.std(ImSemRuido),2))

Q_media = p1_media * p2_media * p3_media

#2 - FILTRO DE LEE

Cmin = 78
Cmax = 95
Lmin = 86
Lmax = 98

varRef = np.var(ImComRuido[Lmin:Lmax,Cmin:Cmax])
varLocal = np.ones((M,N),float)

for i in range(M-dim): #mesmo tamanho utilizado no filtro da média
    for j in range(N-dim):
        varLocal[i+ponto_medio,j+ponto_medio] = np.var(ImComRuido[(i):(i)+dim,(j):(j)+dim])

ImFilt_Lee = np.zeros((M,N),float)
for l in range(M-1):
    for c in range(N-1):
        if varLocal[l,c] == 0:
            k = 1
        else:
            k = 1 - (varRef/varLocal[l,c])

        if k>1:
            k = 1
        elif k<0:
            k = 0
            
        ImFilt_Lee[l,c] = ImFilt_media[l,c] + k*(ImComRuido[l,c] - ImFilt_media[l,c])

plt.figure()
plt.title('Filtro de Lee')
plt.imshow(ImFilt_Lee, cmap='gray')

cv2.destroyAllWindows() ##fechar roi

print('Calculos Filtro de Lee')
Eqmn_Lee = np.math.sqrt(math.pow(np.sum(ImSemRuido-ImFilt_Lee),2)/(N*M))
Emax_Lee = np.max(np.abs(ImSemRuido-ImFilt_Lee))
print('Eqmn_Lee = {} | Emax_Lee = {}'.format(Eqmn_Lee, Emax_Lee))

CoVar_Lee = np.sum((ImSemRuido - np.mean(ImSemRuido)) * (ImFilt_Lee - np.mean(ImFilt_Lee)))/(M*N)

p1_lee = CoVar_Lee/(np.std(ImSemRuido)*np.std(ImFilt_Lee))
p2_lee = (2*(np.mean(ImFilt_Lee) * np.mean(ImSemRuido))) / (math.pow(np.mean(ImFilt_Lee),2) + math.pow(np.mean(ImSemRuido),2))
p3_lee = (2*(np.std(ImFilt_Lee) * np.std(ImSemRuido))) / (math.pow(np.std(ImFilt_Lee),2) + math.pow(np.std(ImSemRuido),2))

Q_Lee = p1_lee * p2_lee * p3_lee
print('Q_Lee = {} | Q_Lee = {}'.format(Q_Lee, Q_Lee))

#3 - FILTRO BUTTERWORTH

fc = 0.15
Do = fc*M/2

ImFreq = np.fft.fft2(ImComRuido)
ImFreqShift = np.fft.fftshift(ImFreq)

H_Butter = np.zeros((M,N),complex)

n=4

for l in range(M):
    for c in range(N):
        distc = c-(N/2) #metade das colunas (N/2 com M/2 formam o valor central)
        distl = l-(M/2) #metade das linhas
        dist = np.sqrt(math.pow(distc,2) + pow(distl,2))
        H_Butter[l,c] = 1/(1+(math.pow(dist/Do,(2*n))))

ImFiltButterFreq = H_Butter*ImFreqShift
ImFilt_Butter = np.abs(np.fft.ifft2(ImFiltButterFreq))

#plot da ImFilt_Butter
plt.figure()
plt.title('Filtro em frequência Butterworth')
plt.imshow(np.abs(ImFilt_Butter), cmap='gray')

print('\n')
print('Calculos Filtro Butterworth')
Eqmn_butter = np.abs(math.pow(np.sum(ImSemRuido-ImFilt_Butter),2)/(N*M))**0.5
Emax_butter = np.max(np.abs(ImSemRuido-ImFilt_Butter))
print('Eqmn_butter = {} | Emax_butter = {}'.format(Eqmn_butter, Emax_butter))

CoVar_butter = np.abs(np.sum((ImSemRuido - np.mean(ImSemRuido)) * (ImFilt_Butter - np.mean(ImFilt_Butter)))/(M*N))

p1_butter = CoVar_butter/(np.std(ImSemRuido)*np.std(ImFilt_Butter))
p2_butter = (2*(np.mean(ImFilt_Butter) * np.mean(ImSemRuido))) / ((np.mean(ImFilt_Butter))**2 + (np.mean(ImSemRuido))**2)
p3_butter = (2*(np.std(ImFilt_Butter) * np.std(ImSemRuido))) / ((np.std(ImFilt_Butter))**2 + (np.std(ImSemRuido))**2)

Q_butter = p1_butter * p2_butter * p3_butter
print('Q_butter = {}'.format(Q_butter))

#4 - FILTRO GAUSSIANO

H_Gauss = np.zeros((M,N),complex)

for l in range(M):
   for c in range(N):
       distc = c-(N/2) #metade das colunas (N/2 com M/2 formam o valor central)
       distl = l-(M/2) #metade das linhas
       dist = np.sqrt(distc**2 + distl**2) #dist=D   
       H_Gauss[l,c] = np.exp(-(dist**2)/(2*(Do**2)))
       
ImFiltGaussFreq = H_Gauss*ImFreqShift
ImFilt_Gauss = np.abs(np.fft.ifft2(ImFiltGaussFreq))

#plot da imagem ImFilt_Gauss
plt.figure()
plt.title('Imagem filtrada Gauss')
plt.imshow(np.abs(ImFilt_Gauss), cmap='gray')

Eqmn_gauss = np.abs((np.sum((ImSemRuido-ImFilt_Gauss)**2)/(N*M))**0.5)
Emax_gauss = np.max(np.abs(ImSemRuido-ImFilt_Gauss))

COV_gauss = np.abs(np.sum((ImSemRuido - np.mean(ImSemRuido)) * (ImFilt_Gauss - np.mean(ImFilt_Gauss)))/(M*N))

p1_gauss = COV_gauss/(np.std(ImSemRuido)*np.std(ImFilt_Gauss))
p2_gauss = (2*(np.mean(ImFilt_Gauss) * np.mean(ImSemRuido))) / ((np.mean(ImFilt_Gauss))**2 + (np.mean(ImSemRuido))**2)
p3_gauss = (2*(np.std(ImFilt_Gauss) * np.std(ImSemRuido))) / ((np.std(ImFilt_Gauss))**2 + (np.std(ImSemRuido))**2)

Q_gauss = p1_gauss * p2_gauss * p3_gauss

#4b - filtro passa baixa ideal 

fc = 0.15

H_ideal = np.zeros((M,N), complex)
Do = fc*(M/2)
    
for l in range(M):
    for c in range(N):
          distc = c-(N/2) #metade das colunas (N/2 com M/2 formam o valor central)
          distl = l-(M/2) #metade das linhas
          dist = np.sqrt(distc**2 + distl**2) #dist=D
          if dist<Do: #condicional da fórmula, analisar gráfico da lista
              H_ideal[l,c] = 1+0j
             
ImFiltIdealFreq = H_ideal*ImFreqShift
ImFilt_Ideal = np.abs(np.fft.ifft2(ImFiltIdealFreq))
              
plt.figure()
plt.title('Imagem filtrada Ideal')
plt.imshow(np.abs(ImFilt_Ideal), cmap='gray')

Eqmn_ideal = np.abs((np.sum((ImSemRuido-ImFilt_Ideal)**2)/(N*M))**0.5)
Emax_ideal = np.max(np.abs(ImSemRuido-ImFilt_Ideal))

COV_ideal = np.abs(np.sum((ImSemRuido - np.mean(ImSemRuido)) * (ImFilt_Ideal - np.mean(ImFilt_Ideal)))/(M*N))

p1_ideal = COV_ideal/(np.std(ImSemRuido)*np.std(ImFilt_Ideal))
p2_ideal = (2*(np.mean(ImFilt_Ideal) * np.mean(ImSemRuido))) / ((np.mean(ImFilt_Ideal))**2 + (np.mean(ImSemRuido))**2)
p3_ideal = (2*(np.std(ImFilt_Ideal) * np.std(ImSemRuido))) / ((np.std(ImFilt_Ideal))**2 + (np.std(ImSemRuido))**2)

Q_ideal = p1_ideal * p2_ideal * p3_ideal

print('Eqmn_ideal = {} | Emax_ideal = {}'.format(Eqmn_ideal, Emax_ideal))
print('Q_ideal = {}'.format(Q_Lee))