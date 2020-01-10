import numpy as np
import matplotlib.pyplot as plt



# This is the results for Shop imbalance Experiments
# We run experiments for 300 epochs and pick the best results.


top_1  = np.array([0.8586, 0.9048, 0.91, 0.9062, 0.9037, 0.8996]) #(DA) 99:39 20,40还没出来
top_10 = np.array([0.9743,0.9809, 0.981, 0.9785, 0.9777, 0.9749]) # (DA) 99:39 20还没出来
top_20 = np.array([0.9834,  0.9873,    0.987,  0.9864,  0.984, 0.9828])  # 00:33 20，40还没出来#SemiHard = np.array([ 20,  40, 80, 160, 320,   480  ,640])  # 00:33 20，40还没出来
top_30 = np.array([0.9876,0.9904, 0.99, 0.9885,0.987, 0.9864])  # 00:33 20，40还没出来
top_40 = np.array([0.9898, 0.9919, 0.991, 0.9900, 0.989, 0.9883])
top_50 = np.array([0.9914, 0.9931, 0.992, 0.9912, 0.9902, 0.9897])
#

b = np.array([640, 960, 1280, 1600 ,1920,2560])



# k_320 = np.array([0.8586, 0.9743, 0.9834, 0.9876, 0.9898, 0.9914])
# k_480 = np.array([0.9048, 0.980, 0.9873, 0.9904, 0.9919, 0.9931])
# k_640 = np.array([0.91, 0.981, 0.987,0.99,0.991, 0.992])
# k_960 = np.array([0.9037, 0.9777, 0.984, 0.987, 0.989, 0.9902])
#
# b = np.array([1, 10, 20, 30, 40, 50])


plt.figure()
plt.plot(b, top_1, 's-',label=r"Recall$@1$", color = plt.cm.YlOrRd(0.9),linewidth=2)
#plt.plot(p, uniform, label=r"uniform", color = 'red')
plt.plot(b, top_10, 's-', label=r"Recall$@10$", color =  plt.cm.YlOrRd(0.8), linewidth=2)
plt.plot(b, top_20, 's-', label=r"Recall$@20$", color = plt.cm.YlOrRd(0.7),linewidth=2)
plt.plot(b, top_30, 's-', label=r"Recall$@30$", color = plt.cm.YlOrRd(0.6), linewidth=2)
plt.plot(b, top_40, 's-', label=r"Recall$@40$", color = plt.cm.YlOrRd(0.5), linewidth=2)
plt.plot(b, top_50, 's-', label=r"Recall$@50$", color = plt.cm.YlOrRd(0.4), linewidth=2)

plt.yticks(fontname="Times New Roman", fontsize=13)
plt.xticks(fontname="Times New Roman", fontsize=13)
plt.legend(loc = 'lower right',fontsize=12)
plt.xlabel(r"K", fontname="Times New Roman", fontsize=17)
plt.ylabel(r"Recall$@$k", fontname="Times New Roman", fontsize=17)
plt.ylim((0.8, 1))
plt.hlines(0.9,500,2750, linestyles='dashed',colors="grey")
#plt.xlim(500, 2750)
plt.xlim(500, 2750)
plt.title(r"Recall$@$k with Different K on In-shop",fontname="Times New Roman", fontsize=17)
#plt.savefig("/Users/qiqi/Desktop/DML-code-results-ICLR/Experiment Results/parametersensitivityintermsofK.eps")
#plt.grid()

plt.show()




# k_320 = np.array([0.8586, 0.9743, 0.9834, 0.9876, 0.9898, 0.9914])
# k_480 = np.array([0.9048, 0.980, 0.9873, 0.9904, 0.9919, 0.9931])
# k_640 = np.array([0.91, 0.981, 0.987,0.99,0.991, 0.992])
# k_960 = np.array([0.9037, 0.9777, 0.984, 0.987, 0.989, 0.9902])
#
# b = np.array([1, 10, 20, 30, 40, 50])
#
#
# plt.figure()
# plt.plot(b, k_320, 'o-',label=r"$k=640$", color = plt.cm.YlOrRd(0.5),linewidth=2)
# #plt.plot(p, uniform, label=r"uniform", color = 'red')
# plt.plot(b, k_480, '^-', label=r"$k=960$", color =  plt.cm.YlOrRd(0.6), linewidth=2)
# plt.plot(b, k_640, 's-', label=r"$k=1280$", color = plt.cm.YlOrRd(0.9),linewidth=2)
# plt.plot(b, k_960, 's-', label=r"$k=1920$", color = plt.cm.YlOrRd(0.7), linewidth=2)
#
# plt.yticks(fontname="Times New Roman", fontsize=12)
# plt.xticks(fontname="Times New Roman", fontsize=12)
# plt.legend(loc = 'lower right')
# plt.xlabel(r"Recall$@$K", fontname="Times New Roman", fontsize=12)
# plt.ylabel(r"Accuracy", fontname="Times New Roman", fontsize=12)
# plt.title(r"Recall with Different K on In-shop",fontname="Times New Roman", fontsize=12)
# plt.savefig("/Users/qiqi/Desktop/DML-code-results-ICLR/Experiment Results/parametersensitivity.eps")
#
# #plt.xlim((0,1))
# plt.ylim((0.8, 1))
# plt.show()
#

