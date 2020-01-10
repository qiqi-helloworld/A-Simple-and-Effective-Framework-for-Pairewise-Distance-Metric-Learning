import numpy as np
import matplotlib.pyplot as plt


MS = np.array([79.8, 94.9, 96.8, 97.6, 97.9, 98.3])
LS = np.array([82.6, 94.1, 95.6, 96.4, 96.9, 97.4])
DRO_1 = np.array([84.8, 95.9, 97.3, 97.9, 98.2, 98.5])
DRO_0_1 = np.array([85.1, 96.1, 97.5, 98.0, 98.3, 98.5])
DRO_00_1 = np.array([85.8, 96.2, 97.9, 97.8, 98.2,98.4])
DRO_000_1 = np.array([85.7, 96.1, 97.4, 97.9, 98.2, 98.5])

V_Recall = np.array([1, 10, 20, 30, 40, 50])
Recall =np.array(range(len(V_Recall)))
#%
plt.figure()
plt.plot(Recall, MS, 'o--',label=r"MS", color = 'green',linewidth=2)
plt.plot(Recall, LS,'^--', label=r"LS", color = 'orange',linewidth=2)
#plt.plot(p, uniform, label=r"uniform", color = 'red')
plt.plot(Recall, DRO_1, 's-', label=r"DRO$+\lambda_p = 1$", color = 'purple', linewidth=2)
plt.plot(Recall, DRO_0_1, 's-', label=r"DRO$+\lambda_p = 0.1$", color = 'yellow',linewidth=2)
plt.plot(Recall, DRO_000_1, 's-', label=r"DRO$+\lambda_p = 0.01$", color = 'blue',linewidth=2)
plt.plot(Recall, DRO_00_1, 's-', label=r"DRO$+\lambda_p = 0.001$", color = 'red',linewidth=2)

plt.xticks(Recall, V_Recall, fontname="Times New Roman")
plt.yticks(fontname="Times New Roman")
#plt.semilogx()

plt.legend(loc = 'best')
plt.xlabel(r"Recall$@$K", fontname="Times New Roman", fontsize=12)
plt.ylabel(r"Accuracy($\%$)",fontname="Times New Roman", fontsize=12)

plt.title(r"Recover of MS and LS Losses",fontname="Times New Roman", fontweight="bold")
#plt.xlim((0,1))
#plt.ylim((0.8, 0.91))
plt.savefig("/Users/qiqi/Desktop/DML-code-results-ICLR/Experiment Results/Inshop_Recover.eps")
plt.show()