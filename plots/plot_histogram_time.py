import numpy as np
import matplotlib.pyplot as plt




x = np.array([0, 160, 320, 480, 640])
topk_A = np.array([0.245, 0.369,	0.528,	0.731, 0.826])
topk_PN = np.array([0.236, 0.398, 0.484, 0.679, 0.687])
PS = np.array([0.235, 0.446, 0.516, 0.667, 0.96])
DWS = np.array([0.549,0.631,0.912,1.221,1.544])
SH = np.array([0.585, 0.689,0.916,1.354,1.788])
MS = np.array([0.439, 0.554,0.889,1.244,1.483])
LS = np.array([0.579, 0.764, 1.021, 1.443, 1.87])

Y=np.array([topk_A, topk_PN, PS, DWS, SH, MS, LS])
labels = [r"DRO-TopK$_M$", r"DRO-TopK-PN$_M$", r'DRO-KL$_M$', r"DWS$_M$", r"SH$_M$", "MS", "LS"]
colors = ["red", "blue","green", "purple", "orange","darkblue","darkred"]

x_0=x

#plt.bar(x, topk_A, color="red", width=barWidth, label=labels[0])
barWidth = 15
plt.figure()
for i in range(7):

    plt.bar(x, Y[i], color=colors[i], width=barWidth, label=labels[i])
    x = x + barWidth
#
plt.xticks((x+x_0-15)/2, [80, 160,320, 480, 640])
plt.xlabel(r"Batchsize", fontname="Times New Roman", fontsize=17)
plt.ylabel(r"Runtime(secs)",fontname="Times New Roman", fontsize=17)
plt.title(r"Average Runtime of Every Iteration",fontname="Times New Roman", fontweight="bold",fontsize=17)
plt.legend(loc = 'best',fontsize = 12)
plt.savefig("/Users/qiqi/Desktop/DML-code-results-ICLR/Experiment Results/runtime_plot.eps")
plt.show()
