#importando as bibliotecas
import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt

#lendo a imagem
IVUSRef = cv2.imread('IVUSReferencia.pgm',0)
IVUSRef = skimage.img_as_float(IVUSRef)

#plot da imagem
plt.figure()
plt.title('IVUS Referencia')
plt.imshow(IVUSRef,cmap='gray')

cv2.destroyAllWindows() ##fechar roi

#ROI
IVUSRef_ROI = cv2.selectROI(IVUSRef)

Cmin = IVUSRef_ROI[0]
Lmin = IVUSRef_ROI[1] 
Cmax = IVUSRef_ROI[0]+IVUSRef_ROI[2]
Lmax = IVUSRef_ROI[1]+IVUSRef_ROI[3]

#calculo da media e desvio padrao da regiao selecionada
media = np.mean(IVUSRef[Lmin:Lmax,Cmin:Cmax])
desvpad = np.std(IVUSRef[Lmin:Lmax,Cmin:Cmax])

print('Media: {} | Desvio Padrao: {}'.format(media,desvpad))

#verificando quais pixels possuem a mesma textura da regiao seecionada
(M,N) = np.shape(IVUSRef)

Obj = np.zeros((M,N))
for l in range(M):
    for c in range (N):
        if((IVUSRef[l,c] >= (media - 0.5*desvpad)) & (IVUSRef[l,c] <= (media - 0.5*desvpad))):
           Obj[l,c] = IVUSRef[l,c]

plt.figure()
plt.title('Segmentacao')
plt.imshow(Obj,cmap='gray')

#gerando 5 imagens com ruidos
ImComRuido_1 = skimage.util.random_noise(IVUSRef, mode='gaussian', seed=None, clip=True, mean=0, var=0.001)
ImComRuido_2 = skimage.util.random_noise(IVUSRef, mode='gaussian', seed=None, clip=True, mean=0, var=0.002)
ImComRuido_3 = skimage.util.random_noise(IVUSRef, mode='gaussian', seed=None, clip=True, mean=0, var=0.003)
ImComRuido_4 = skimage.util.random_noise(IVUSRef, mode='gaussian', seed=None, clip=True, mean=0, var=0.004)
ImComRuido_5 = skimage.util.random_noise(IVUSRef, mode='gaussian', seed=None, clip=True, mean=0, var=0.005)

#plot das figuras com ruidos
plt.figure()
plt.title('IVUSRef com ruido 0.001')
plt.imshow(ImComRuido_1, cmap='gray')

plt.figure()
plt.title('IVUSRef com ruido 0.002')
plt.imshow(ImComRuido_2, cmap='gray')

plt.figure()
plt.title('IVUSRef com ruido 0.003')
plt.imshow(ImComRuido_3, cmap='gray')

plt.figure()
plt.title('IVUSRef com ruido 0.004')
plt.imshow(ImComRuido_4, cmap='gray')

plt.figure()
plt.title('IVUSRef com ruido 0.005')
plt.imshow(ImComRuido_5, cmap='gray')

#fazendo a segmentação para cada imagem
import bibSegmentacao

obj_1 = bibSegmentacao.fazerSegmentacao(ImComRuido_1)
plt.figure()
plt.title('Obj_1')
plt.imshow(obj_1, cmap='gray')

obj_2 = bibSegmentacao.fazerSegmentacao(ImComRuido_2)
plt.figure()
plt.title('Obj_2')
plt.imshow(obj_2, cmap='gray')

obj_3 = bibSegmentacao.fazerSegmentacao(ImComRuido_3)
plt.figure()
plt.title('Obj_3')
plt.imshow(obj_3, cmap='gray')

obj_4 = bibSegmentacao.fazerSegmentacao(ImComRuido_4)
plt.figure()
plt.title('Obj_4')
plt.imshow(obj_4, cmap='gray')

obj_5 = bibSegmentacao.fazerSegmentacao(ImComRuido_5)
plt.figure()
plt.title('Obj_5')
plt.imshow(obj_5, cmap='gray')

#Lendo a imagem Gold Standard
GoldStandard = cv2.imread('ObjetoGoldStandard.pgm',0)
GoldStandard = skimage.img_as_float(GoldStandard)

plt.figure()
plt.title('ObjetoGoldStandard')
plt.imshow(GoldStandard,cmap='gray')

#binarizando a imagem gold standard
thresh = 0.5
ObjetoGoldStandardBin = GoldStandard > thresh

plt.figure()
plt.title('ObjetoGoldStandardBin')
plt.imshow(ObjetoGoldStandardBin,cmap='gray')

#Avaliando para as 5 imagens
import Avaliacao

aval_1 = Avaliacao.fazerAvaliacaoSegmentacao(obj_1, ObjetoGoldStandardBin)
aval_2 = Avaliacao.fazerAvaliacaoSegmentacao(obj_2, ObjetoGoldStandardBin)
aval_3 = Avaliacao.fazerAvaliacaoSegmentacao(obj_3, ObjetoGoldStandardBin)
aval_4 = Avaliacao.fazerAvaliacaoSegmentacao(obj_4, ObjetoGoldStandardBin)
aval_5 = Avaliacao.fazerAvaliacaoSegmentacao(obj_5, ObjetoGoldStandardBin)

vetor = np.zeros(5)

for i in range(0,3):
    vetor[0] = aval_1[i]
    vetor[1] = aval_2[i]
    vetor[2] = aval_3[i]
    vetor[3] = aval_4[i]
    vetor[4] = aval_5[i]
    
    media = np.mean(vetor)
    desvpad = np.std(vetor)
    
    print('Média {}: {}'.format(i+1, media))
    print('Desvio Padrão {}: {}'.format(i+1, desvpad))
    print('\n')

