def fazerSegmentacao(Img):
    
    #importando as bibliotecas
    import numpy as np
    import cv2
    import skimage
    import matplotlib.pyplot as plt
    import scypi.signal
    
    mascara = 5
    w = np.ones((mascara,mascara), float)/(mascara*mascara)
    
    Img = scypi.signal.convolved2d(Img,w,'same')
    
    #ROI
    Img_ROI = cv2.selectROI(Img)
    
    Cmin = Img_ROI[0]
    Lmin = Img_ROI[1] 
    Cmax = Img_ROI[0]+Img_ROI[2]
    Lmax = Img_ROI[1]+Img_ROI[3]
    
    #calculo da media e desvio padrao da regiao selecionada
    media = np.mean(Img[Lmin:Lmax,Cmin:Cmax])
    desvpad = np.std(Img[Lmin:Lmax,Cmin:Cmax])
    
    #verificando quais pixels possuem a mesma textura da regiao seecionada
    (M,N) = np.shape(Img)
    
    Obj = np.zeros((M,N))
    for l in range(M):
         for c in range (N):
                if((Img[l,c] >= (media - 0.5*desvpad)) & (Img[l,c] <= (media + 0.5*desvpad))):
                   Obj[l,c] = Img[l,c]
    return Obj