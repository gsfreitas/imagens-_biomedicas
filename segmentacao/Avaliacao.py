def fazerAvaliacaoSegmentacao(ObjetoSegmentado, GoldStandard):
    import numpy as np
    import cv2
    import skimage
    import matplotlib.pyplot as plt
    
    Intersection = ObjetoSegmentado * GoldStandard
    A_Intersection = np.sum(Intersection)
    A_Segmentada = np.sum(ObjetoSegmentado)
    A_GoldStandard = np.sum(GoldStandard)
    
    (M,N) = np.shape(ObjetoSegmentado)
    
    VP = (A_Intersection/A_GoldStandard)*100
    FP = ((A_Segmentada - A_Intersection)/(M*N - A_GoldStandard))*100
    FN = ((A_GoldStandard - A_Intersection)/A_GoldStandard)*100
    
    resultado = [VP, FP, FN]
    
    return resultado