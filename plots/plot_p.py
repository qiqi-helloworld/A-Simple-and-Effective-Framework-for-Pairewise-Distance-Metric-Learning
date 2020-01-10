import numpy as np
import matplotlib.pyplot as plt



top_k = np.array([0.593, 0.625, 0.674, 0.663, 0.6425])
uniform = np.array([0.591, 0.629, 0.6604, 0.658, 0.5265])
semihard = np.array([0.591, 0.6423, 0.657, 0.6702, 0.6681])
x = np.array([20, 40, 80, 160, 320])
p = np.array([0.257, 0.114, 0.053, 0.026, 0.012])

plt.figure()
plt.plot(p, top_k, label=r"top-k", color = 'orange')
plt.plot(p, uniform, label=r"uniform", color = 'red')
plt.plot(p, semihard, label=r"semihard", color = 'blue')

plt.legend(loc = 'best')
plt.xlabel(r"ratio of positive pairs and negative pairs")
plt.ylabel(r"$recall@1$")
plt.title(r".")
plt.title(r"Different mining methods with different imbalance ratio")
#plt.xlim((0,1))
plt.ylim((0.3, 0.7))
plt.show()




plt.figure()
plt.plot(x, top_k, label=r"top-k", color = 'orange')
plt.plot(x, uniform, label=r"uniform", color = 'red')
plt.plot(x, semihard, label=r"semihard", color = 'blue')
plt.legend(loc = 'best')
plt.xlabel(r"$Batch Size$")
plt.ylabel(r"$recall@1$")
plt.title(r".")
plt.title(r"Comparing Different Mining Methods")
#plt.xlim((0,1))
plt.ylim((0.3, 0.7))
plt.show()


#x_0 = np.zeros(1000)
#x= x_0
# #print(x)
#
# x = np.array(range(-100, 100, 1))/100
# print(x)
# pos_loss = np.maximum((x - 0.3), 0)
# neg_loss = np.maximum((0.7 - x), 0)
#
# lamda = [0.1, 0.2, 1]
# print(np.sum(np.exp(pos_loss/lamda[0])))
# p1 = (np.exp(pos_loss/lamda[0]) - 1)/(np.sum(np.exp(pos_loss/lamda[0]) - 1))
# p2 = (np.exp(pos_loss/lamda[1]) - 1)/(np.sum(np.exp(pos_loss/lamda[1]) - 1))
# p3 = (np.exp(pos_loss/lamda[2]) - 1)/(np.sum(np.exp(pos_loss/lamda[2]) - 1))
# n1 = (np.exp(neg_loss/lamda[0]) -1)/(np.sum(np.exp(neg_loss/lamda[0]) - 1))
# n2 = (np.exp(neg_loss/lamda[1]) -1)/(np.sum(np.exp(neg_loss/lamda[1]) - 1))
# n3 = (np.exp(neg_loss/lamda[2]) -1)/(np.sum(np.exp(neg_loss/lamda[2]) - 1))
#
# plt.figure()
# plt.plot(x, p1, label=r"$y_{ij} = 1, \lambda=0.1$", color = 'orange')
# plt.plot(x, p2, label=r"$y_{ij} = 1, \lambda=0.2$", color = 'red')
# plt.plot(x, p3, label=r"$y_{ij} = 1, \lambda= 1$", color = 'green')
# plt.plot(x, n1, label=r"$y_{ij} = -1, \lambda=0.1$", linestyle = '--', color = 'orange')
# plt.plot(x, n2, label=r"$y_{ij} = -1, \lambda=0.2$", linestyle = '--', color = 'red')
# plt.plot(x, n3, label=r"$y_{ij} = -1, \lambda=1$", linestyle = '--',color = 'green')
# plt.legend(loc = 'best')
# plt.xlabel(r"$S_{ij}$")
# plt.ylabel(r"$p$")
# plt.title(r".")
# plt.title(r"$\ell(S_{ij}, y_{ij}) = (0.2 + y_{ij}(\lambda - 0.5))_+$")
# plt.ylim((0,0.1))
# plt.savefig("../Experiment Results/p_over_S_by_class.eps")
#
#
#
# #x_0 = np.zeros(1000)
# #x= x_0
# #print(x)
#
# x = np.array(range(-0, 100, 1))/100
# print(x)
# pos_loss = np.maximum((x - 0.3), 0)
# neg_loss = np.maximum((0.7 - x), 0)
#
# lamda = [0.1, 0.2, 1]
# print(np.sum(np.exp(pos_loss/lamda[0])))
# p1 = (np.exp(pos_loss/lamda[0]) - 1)/(np.sum(np.exp(pos_loss/lamda[0]) - 1) + (np.sum(np.exp(neg_loss/lamda[0]) - 1)))
# p2 = (np.exp(pos_loss/lamda[1]) - 1)/(np.sum(np.exp(pos_loss/lamda[1]) - 1) + (np.sum(np.exp(neg_loss/lamda[1]) - 1)))
# p3 = (np.exp(pos_loss/lamda[2]) - 1)/(np.sum(np.exp(pos_loss/lamda[2]) - 1) + (np.sum(np.exp(neg_loss/lamda[2]) - 1)))
# n1 = (np.exp(neg_loss/lamda[0]) -1)/(np.sum(np.exp(pos_loss/lamda[0]) - 1) + (np.sum(np.exp(neg_loss/lamda[0]) - 1)))
# n2 = (np.exp(neg_loss/lamda[1]) -1)/(np.sum(np.exp(pos_loss/lamda[1]) - 1) + (np.sum(np.exp(neg_loss/lamda[1]) - 1)))
# n3 = (np.exp(neg_loss/lamda[2]) -1)/(np.sum(np.exp(pos_loss/lamda[2]) - 1) + (np.sum(np.exp(neg_loss/lamda[2]) - 1)))
#
# plt.figure()
# plt.plot(x, p1, label=r"$y_{ij} = 1, \lambda=0.1$", color = 'orange')
# plt.plot(x, p2, label=r"$y_{ij} = 1, \lambda=0.2$", color = 'red')
# plt.plot(x, p3, label=r"$y_{ij} = 1, \lambda= 1$", color = 'green')
# plt.plot(x, n1, label=r"$y_{ij} = -1, \lambda=0.1$", linestyle = '--', color = 'orange')
# plt.plot(x, n2, label=r"$y_{ij} = -1, \lambda=0.2$", linestyle = '--', color = 'red')
# plt.plot(x, n3, label=r"$y_{ij} = -1, \lambda=1$", linestyle = '--',color = 'green')
# plt.legend(loc = 'best')
# plt.xlabel(r"$S_{ij}$")
# plt.ylabel(r"$p$")
# plt.title(r".")
# plt.title(r"$\ell(S_{ij}, y_{ij}) = (0.2 + y_{ij}(\lambda - 0.5))_+$")
# #plt.xlim((0,1))
# plt.ylim((0, 0.06))
#
# plt.savefig("../Experiment Results/p_over_S_over_batch.eps")
# plt.show()