# importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure

# lendo as imagens
Mamo = cv2.imread('Mamography.pgm', 0)
Stent = cv2.imread('Stent.pgm',0)

# normalizando as imagens
MamoN = skimage.img_as_float(Mamo)
StentN = skimage.img_as_float(Stent)

# exibindo as imagens da mamografia
plt.figure()
plt.ylabel('Linhas - M')
plt.xlabel('Colunas - N')
plt.title('MamoN')
plt.imshow(MamoN, cmap='gray') # cmap='jet'
plt.colorbar()

# obtendo o número de linhas e colunas da imagem mamo
M, N = np.shape(Mamo)

# criando matriz para o negativo de Mamo
NegativoMamoPorPixel = np.zeros((M,N), float)
NegativoMamoDireto = np.zeros((M,N), float)


# varrendo a matriz e criando a f[ormula para o negativo da imagem Mamo
for l in range(M):
    for c in range(N):
        NegativoMamoPorPixel [l,c] = 255-Mamo[l,c]

# plot do negativo da imagem
plt.figure()
plt.ylabel('Linhas - M')
plt.xlabel('Colunas - N')
plt.title('NegativoMamoPorPixel')
plt.imshow(NegativoMamoPorPixel , cmap='gray') # cmap='jet'
plt.colorbar()

# criando o negativo da imagem mamo de forma direta, sem o loop
NegativoMamoDireto = 255-Mamo

# plot do negativo de forma direta
plt.figure()
plt.ylabel('Linhas - M')
plt.xlabel('Colunas - N')
plt.title('NegativoMamoDireto')
plt.imshow(NegativoMamoDireto , cmap='gray') # cmap='jet'
plt.colorbar()

# cria o histograma de intensidades da imagem mamo
import bibFuncoesHistograma
histograma = bibFuncoesHistograma.fazerHistograma(Mamo, M, N)

# plot do histograma
plt.figure()
plt.stem(histograma, use_line_collection=True)
plt.title('Histograma 1')
plt.xlabel('Classes')
plt.ylabel('Numero de Ocorrência')

plt.show()

# analisando e comparando com o segundo histograma
histograma2 = skimage.exposure.histogram(Mamo)
x = histograma2[1] # Classes
y = histograma2[0] # Numero de Ocorrência
plt.figure()
plt.title('Histograma 2')
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.stem(histograma2, use_line_collection=True)
plt.show()

# Refaça o exercício anterior, mas agora para as classes da imagem
# normalizada entre 0 e 1, chame de histograma3. Verifique se as classes
# ficaram entre 0 e 1.
histograma3 = skimage.exposure.histogram(MamoN)
x = histograma3[1] # Classes
y = histograma3[0] # Numero de Ocorrência
plt.figure()
plt.stem(x, y, use_line_collection=True)
plt.ylabel('Numero de Ocorrência')
plt.xlabel('Classes')
plt.stem(x,y, use_line_collection=True)
plt.title('Histograma 3')
plt.show()

# faça uma operação que aumente o nível de cinza(brilho) da imagem
# “StentN” em mais 0.2 níveis de intensidade, considerando ela entre 0 - 1.
StentBrilhoN = StentN+0.2

# renormalizar intensidade para não sair do range, incluir níveis de intensidades
# fora do rage
StentBrilhoN = skimage.exposure.rescale_intensity(StentBrilhoN, in_range=(0,1))

# Ajuste de Contraste
StentAlongada = skimage.exposure.rescale_intensity(StentBrilhoN, in_range=(0.2,0.7))

# plot da stent alongada
plt.figure()
plt.title('Stent Alongada')
plt.imshow(StentAlongada)

