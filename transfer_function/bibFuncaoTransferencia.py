def fazerMascaraIdeal(M, N, fc):
    import numpy as np
    
    #criando uma matriz de zeros
    
    H_Ideal = np.zeros((M,N),complex)
    
    
    Do = fc*(M/2)
    
    for l in range(M):
        for c in range(N):
            dist_x = c - (N/2)
            dist_y = l - (M/2)
            
            D = np.math.sqrt(pow(dist_x,2)+pow(dist_y,2))
            
            if(D < Do):
                H_Ideal[l,c] = 1 + 0j
    
    return H_Ideal

def fazerMascaraGaussiana2D(M, N, fc):
    import scipy.signal
    import numpy as np
    import cv2
    
    #criando uma matriz de zeros
    H_Gauss = np.zeros((M,N), complex)
    
    Do = fc*(M/2)
    
    for l in range(M):
        for c in range(N):
            dist_x = c - (M/2)
            dist_y = l - (N/2)
            
            D = np.math.sqrt(pow(dist_x, 2)+pow(dist_y,2))
            
            H_Gauss[l,c] = np.exp(-pow(D,2)/(2*Do))
    
    return H_Gauss

def fazerMascaraButterworth(M,N,fc,n):
    import scipy.signal
    import numpy as np
    import cv2
    
    H_Butter = np.zeros((M,N), complex)
    
    Do = fc*(M/2)
    
    for l in range(M):
        for c in range(N):
            dist_x = c - (M/2)
            dist_y = l - (N/2)
            
            D = np.math.sqrt(pow(dist_x, 2)+pow(dist_y,2))
            
            H_Butter[l,c] = 1/(1+pow(pow(D/Do,2),2*n))
    
    return H_Butter