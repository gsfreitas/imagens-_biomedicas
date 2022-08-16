def fazerMascaraGauss2D(media, desvio):
    import scipy.signal
    import numpy as np
    import cv2
    
    pos = 2*media + 1
    G = scipy.signal.gaussian(pos, desvio)
    
    #criando a máscara
    G1 = np.zeros((pos,pos), float)
    G1[media,:] = G
    Gtranspose1 = np.transpose(G1)
    w_Gauss2D = scipy.signal.convolve2d(G1,Gtranspose1,'same')
    w_Gauss2DNormal = w_Gauss2D/(np.sum(w_Gauss2D))
    
    return w_Gauss2D