import numpy as np
import matplotlib.pyplot as plt



# This is the results for Shop imbalance Experiments
# We run experiments for 300 epochs and pick the best results.

def back_up_plot():

    # Top-K: DIM-1024-DA
    # Top-K-by-Class: DIM-1024-DA
    # Semihard: DIM-512
    # DWS: DIM-512
    # If you need to find the results, just look at 33 and 39 server.

    top_k = np.array([0.8064, 0.8064, 0.8612, 0.8890, 0.8967, 0.9046]) #(DA) 99:39 20,40还没出来
    top_k_by_class = np.array([0.8064,0.8064, 0.8859,0.8985,0.907, 0.9086]) # (DA) 99:39 20还没出来
    semihard = np.array([0.8064, 0.8467, 0.8753, 0.8847, 0.8934, 0.8976]) #Fair, set positive pairs similarity equal to threshold
    DWS = np.array([0.8064, 0.8064, 0.8822, 0.8877, 0.8879, 0.8766]) #00:33 20，40还没出来

    #uniform = np.array([0]) #00:33
    #DRO = np.array([*, *, *, *, *, *])
    b = np.array([20, 40, 80, 160, 320, 640])
    p = np.array([0.257, 0.114, 0.053, 0.026, 0.012, 0.006])

    # plt.figure()
    # plt.plot(p, top_k, label=r"top-k", color = 'red')
    # #plt.plot(p, uniform, label=r"uniform", color = 'red')
    # plt.plot(p, semihard, label=r"semihard", color = 'blue')
    # plt.plot(p, DWS, label=r"DWS", color = 'orange')
    #
    # plt.legend(loc = 'best')
    # plt.xlabel(r"ratio of positive pairs and negative pairs")
    # plt.ylabel(r"$recall@1$")
    # plt.title(r".")
    # plt.title(r"Different mining methods with different imbalance ratio")
    # #plt.xlim((0,1))
    # plt.ylim((0.7, 0.9))
    # plt.show()

    plt.figure()
    plt.plot(b[2:], top_k[2:], 's-',label=r"top-k:DIM-1024-DA", color = 'red')
    plt.plot(b[2:], top_k_by_class[2:],'s-', label=r"top-k-by-class:DIM-1024-DA", color = 'blue')

    #plt.plot(p, uniform, label=r"uniform", color = 'red')
    plt.plot(b[2:], semihard[2:], 'o-', label=r"SemiHard:DIM-512", color = 'green', linewidth=2)
    plt.plot(b[2:], DWS[2:], '^-', label=r"DWS:DIM-512", color = 'orange')

    plt.legend(loc = 'best')
    plt.xlabel(r"batchsize")
    plt.ylabel(r"$recall@1$")
    plt.title(r".")
    plt.title(r"Imbalance Resisting With Different Batchsize")
    #plt.xlim((0,1))
    plt.ylim((0.8, 0.91))
    plt.show()


#back_up_plot() #not use in the end
def dim_1024_imbalance():

    top_k = np.array([0.8064, 0.8064, 0.8612, 0.8890, 0.8967, 0.9039, 0.9046]) #(DA) 99:39 20,40还没出来
    top_k_by_class = np.array([0.8064,0.8064, 0.8859, 0.8985,0.907, 0.9086, 0.9086]) # (DA) 99:39 20还没出来
    DWS =      np.array([20,  40,      0.8798,       0.8852,      0.8689,    0.8519,    0.8319])  # 00:33 20，40还没出来#SemiHard = np.array([ 20,  40, 80, 160, 320,   480  ,640])  # 00:33 20，40还没出来
    semihard = np.array([ 20,  40, 0.8689, 0.8855, 0.8998,   0.8991,  0.9005])  # 00:33 20，40还没出来
    DRO =      np.array([20,  40, 0.8829,   0.8869,  0.8982,  0.8998,  0.9018])  # 00:33 20，40还没出来#SemiHard = np.array([ 20,  40, 80, 160, 320,   480  ,640])  # 00:33 20，40还没出来


    b = np.array([20, 40, 80, 160, 320, 480, 640])
    newp = np.array([0.257, 0.114, 0.053, 0.026, 0.012, 0.008, 0.006])
    p = np.array(range(len(newp)))

    indices= [2,3,4,5, 6]
    print(p[indices])
    plt.figure()
    plt.plot(p[indices], DWS[indices], '^-', label=r"DWS$_M$", color = 'orange',linewidth=3)
    plt.plot(p[indices], semihard[indices], 'o-', label=r"SH$_M$", color = 'purple', linewidth=3)
    plt.plot(p[indices], top_k[indices], 's-',label=r"DRO-TopK$_M$", color = 'red',linewidth=3)
    plt.plot(p[indices], top_k_by_class[indices],'s-', label=r"DRO-TopK-PN$_M$", color = 'blue',linewidth=3)
    plt.plot(p[indices], DRO[indices], 's-', label=r'DRO-KL$_M$', color= 'green',linewidth=3)
    plt.xticks(p[indices], newp[indices], fontname="Times New Roman", fontsize= 17)
    plt.yticks(fontname="Times New Roman", fontsize= 17)
    #plt.semilogx()
    #=
    plt.legend(loc = 'best', fontsize = 15)
    plt.xlabel(r"P-N Ratio", fontname="Times New Roman", fontsize=17)
    plt.ylabel(r"Recall$@$1",fontname="Times New Roman", fontsize=17)
    plt.title(r"Recall$@$k on Different P-N Ratio",fontname="Times New Roman", fontweight="bold", fontsize=17)
    #plt.xlim((0,1))
    plt.ylim((0.8, 0.91))
    plt.savefig("/Users/qiqi/Desktop/DML-code-results-ICLR/Experiment Results/Imbalance_Ratio_1.eps")
    plt.show()


def dim_512_imbalance():
    top_k = np.array([0.887, 0.8972, 0.9035, 0.9062, 0.9058])  # (DA) 99:39 20,40还没出来
    top_k_by_class = np.array([0.8822, 0.8975, 0.9052, 0.9045, 0.9073])  # (DA) 99:39 20还没出来
    DWS = np.array([0.8857, 0.8904, 0.8938, 0.8911, 0.8882])  # 0.8977，40还没出来#SemiHard = np.array([ 20,  40, 80, 160, 320,   480  ,640])  # 00:33 20，40还没出来
    semihard = np.array([0.8712, 0.8870, 0.8950, 0.8962, 0.9010])  # 00:33 20，40还没出来
    DRO = np.array([0.8916, 0.8939,0.8972,0.9013, 0.9021])  # 00:33 20，40还没出来#SemiHard = np.array([ 20,  40, 80, 160, 320,   480  ,640])  # 00:33 20，40还没出来
    newp = np.array([0.053, 0.026, 0.012, 0.008, 0.006])
    p = np.array(range(len(newp)))

    indices = [0,1,2,3,4]
    print(p[indices])
    plt.figure()
    plt.plot(p[indices], DWS[indices], '^-', label=r"DWS$_M$", color='orange', linewidth=3)
    plt.plot(p[indices], semihard[indices], 'o-', label=r"SH$_M$", color='purple', linewidth=3)
    plt.plot(p[indices], top_k[indices], 's-', label=r"DRO-TopK$_M$", color='red', linewidth=3)
    plt.plot(p[indices], top_k_by_class[indices], 's-', label=r"DRO-TopK-PN$_M$", color='blue', linewidth=3)
    plt.plot(p[indices], DRO[indices], 's-', label=r'DRO-KL$_M$', color='green', linewidth=3)
    plt.xticks(p[indices], newp[indices], fontname="Times New Roman", fontsize=17)
    plt.yticks(fontname="Times New Roman", fontsize=17)
    # plt.semilogx()
    # =
    plt.legend(loc='best', fontsize=15)
    plt.xlabel(r"P-N Ratio", fontname="Times New Roman", fontsize=17)
    plt.ylabel(r"Recall$@$1", fontname="Times New Roman", fontsize=17)
    plt.title(r"Recall$@$k on Different P-N Ratio", fontname="Times New Roman", fontweight="bold", fontsize=17)
    # plt.xlim((0,1))
    plt.ylim((0.8, 0.91))
    plt.savefig("/Users/qiqi/Documents/File/a-research/2020-ICLR/Experiment Results/Imbalance_Ratio_512.eps")
    plt.show()

if __name__ == '__main__':
    dim_512_imbalance()
    print('Congratulations to you!')

