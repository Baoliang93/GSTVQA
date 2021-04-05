#!/etc/bin/python
#coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


yLQ = [[-0.0211, -0.0148, -0.0201,  0.0205, -0.0207, -0.0206, -0.0281,  0.0212,
        -0.0214,  0.0200, -0.0111, -0.0097,  0.0280,  0.0292, -0.0212, -0.0200,
         0.0187, -0.0276,  0.0155,  0.0184,  0.0211, -0.0177, -0.0202,  0.0176,
        -0.0167, -0.0236,  0.0197,  0.0152,  0.0181,  0.0198,  0.0144, -0.0169],[1.0113, 0.9800, 1.0422, 1.0411, 1.0423, 1.0422, 0.9684, 0.9745, 1.0422,
        1.0416, 1.0413, 1.0175, 0.9872, 0.9752, 0.9792, 1.0125, 0.9481, 0.9483,
        0.9857, 0.9663, 1.0445, 0.9467, 1.0427, 1.0414, 1.0406, 0.9951, 0.9791,
        0.9775, 0.9719, 0.9451, 1.0426, 1.0147]]

yCVD =[[-0.0418,  0.0280,  0.0280,  0.0451,  0.0263,  0.0255, -0.0432,  0.0416,
         0.0265, -0.0201,  0.0318, -0.0351, -0.0202, -0.0212,  0.0160,  0.0120,
        -0.0264,  0.0303, -0.0273, -0.0380,  0.0067, -0.0421,  0.0287,  0.0342,
        -0.0419,  0.0288, -0.0283,  0.0322,  0.0367,  0.0447, -0.0286,  0.0238],[0.9215, 0.9407, 0.9459, 0.9289, 0.9467, 0.9486, 0.9157, 0.9310, 0.9495,
        0.9514, 0.9512, 0.9224, 0.9481, 0.9534, 0.9533, 0.9457, 0.9365, 0.9308,
        0.9491, 0.9268, 0.9847, 0.9113, 0.9504, 0.9324, 0.9537, 0.9488, 0.9330,
        0.9295, 0.9461, 0.9165, 0.9484, 0.9406]]

yLV = [[-0.0640, -0.0532, -0.0282,  0.0340, -0.0367, -0.0327, -0.0725,  0.0521,
        -0.0733,  0.0343,  0.0016,  0.0224,  0.0721,  0.0643, -0.0559, -0.0295,
         0.0673, -0.0818,  0.0457,  0.0559,  0.0766, -0.0727, -0.0441,  0.0187,
        -0.0234, -0.0596,  0.0493,  0.0483,  0.0499,  0.0579,  0.0237, -0.0018],[0.9523, 0.8880, 1.0830, 1.0630, 1.0827, 1.0831, 0.8888, 0.8883, 0.9092,
        1.0832, 1.0135, 0.9343, 0.9159, 0.8946, 0.9001, 1.0154, 0.8501, 0.8435,
        0.9092, 0.8705, 1.0790, 0.8504, 1.0791, 1.0825, 1.0811, 0.9266, 0.9020,
        0.9162, 0.8958, 0.8668, 1.0964, 0.9744]]

yKON = [[0.0128, -0.0593, -0.0455,  0.0490, -0.0461, -0.0464, -0.0574,  0.0731,
        -0.0995,  0.0481,  0.0088, -0.0973,  0.0161,  0.0942, -0.0638, -0.0204,
         0.0868, -0.0865,  0.0664,  0.0607,  0.0499, -0.1065, -0.0567,  0.0296,
        -0.0334, -0.1400,  0.0815,  0.0736,  0.0820,  0.0803,  0.0366, -0.0018],[0.9843, 0.8574, 1.1154, 1.1146, 1.1128, 1.1148, 0.8515, 0.8512, 0.8478,
        1.1165, 0.9922, 0.8463, 1.0359, 0.8542, 0.8796, 1.0092, 0.8014, 0.8173,
        0.8638, 0.8594, 1.1282, 0.7689, 1.1184, 1.1150, 1.1163, 1.0489, 0.8627,
        0.8736, 0.8504, 0.8209, 1.1178, 0.9928]]
GD0 = np.zeros(32)
GD1 = GD0+1
fonten = {'family':'Times New Roman','size': 11,'weight':'roman'}

for i in range(1,5):
    if i==1: 
        y_major_locator=MultipleLocator(0.01)
        yyLQ= yCVD
        name = 'CVD2014'
    if i==2: 
        y_major_locator=MultipleLocator(0.01)
        yyLQ= yLQ
        name = 'LIVE-Qualcomn'
    if i==3: 
        y_major_locator=MultipleLocator(0.02)
        yyLQ= yLV
        name = ' LIVE-VQC'
    if i==4: 
        y_major_locator=MultipleLocator(0.02)
        yyLQ= yKON
        name = 'KoNViD-1k'
    #np.random.seed(2000)
    #y = np.random.standard_normal((10, 2))
    x_major_locator=MultipleLocator(2)
    
    plt.figure(figsize=(7,5))
    plt.subplot(211)  #两行一列,第一个图
    #plt.plot(yLQ[0], lw = 1.5,label = 'LIVE-Q')
    plt.plot(yyLQ[0], 'k.',markersize=5.5)
    plt.plot(GD0,'r--',linewidth=1.2)
    plt.yticks(fontproperties = 'Times New Roman', size = 10)
    plt.xticks(fontproperties = 'Times New Roman', size = 10)
    plt.grid(axis="y",linestyle='--')
    #plt.grid(axis="x",linestyle='--')
    #plt.legend(loc = 1) #图例位置自动
    plt.axis('tight')
    plt.ylabel('Mean', fontdict=fonten)
#    plt.title('Trained on '+name+'dataset', fontdict=fonten)

    plt.subplot(212) #两行一列.第二个图
    #plt.plot(yLQ[1],'g', lw = 1.5,label = 'LIVE-Q')
    plt.plot(yyLQ[1], 'k^',markersize=3.5)
    plt.plot(GD1,'r--',linewidth=1.2)
    plt.yticks(fontproperties = 'Times New Roman', size = 10)
    plt.xticks(fontproperties = 'Times New Roman', size = 10)
    #plt.grid(axis="x",linestyle='--')
    plt.grid(axis="y",linestyle='--')
    #plt.legend(loc = 1)
    plt.xlabel('Index', fontdict=fonten)
    plt.ylabel('Variance', fontdict=fonten)
    plt.axis('tight')
    
#    ax=plt.gca()    
#    ax.xaxis.set_major_locator(x_major_locator)
#    ax.yaxis.set_major_locator(y_major_locator)
    
    plt.savefig(name+"MandV.png", dpi=1200)
    plt.show()