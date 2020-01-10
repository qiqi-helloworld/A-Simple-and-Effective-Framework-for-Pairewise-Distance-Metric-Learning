import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



top_k_A = np.array([[0.6745, 0.7772, 0.8589, 0.9156, 0.9541, 0.9777],
                 [0.6788, 0.7812, 0.8584, 0.9156, 0.9517, 0.9747],
                 [0.6761, 0.7768, 0.8671, 0.9127, 0.9519, 0.9723],
                    [0.6761,0.7785, 0.8604, 0.9142, 0.9522, 0.9732]
                    ]) # Data Augmentation

top_k_PN = np.array([[0.6702, 0.7772, 0.8648, 0.9198, 0.9553, 0.9764],
                 [0.6735, 0.7708, 0.8587, 0.9156, 0.9502, 0.9747],
                 [0.6732, 0.7755, 0.8569, 0.9127, 0.9475, 0.9698]]) # Data Augmentation


binomial_A = np.array([[0.6833, 0.7846, 0.8582, 0.9149, 0.9527, 0.9754],
                 [0.6767, 0.7812, 0.8550, 0.9117, 0.9529, 0.9743],
                 [0.6764, 0.7768, 0.8589, 0.9156, 0.9534, 0.9748]]) # No data Augmentation

binomial_PN = np.array([[0.6771, 0.7806, 0.8575, 0.9121, 0.9522, 0.9738],
                        [0.6757, 0.7790, 0.8545, 0.9104, 0.9465, 0.9711],
                        [0.6745, 0.7714, 0.8558, 0.9117, 0.9499, 0.9740],
                        [0.6796, 0.7836, 0.8661, 0.9173, 0.9512, 0.9728],
                        [0.6783, 0.7794, 0.8558, 0.9102, 0.9472, 0.9706]]) # data Augmentation

top_k_KL = np.array([[0.6771, 0.7854, 0.8628, 0.9205, 0.9563, 0.9750],
                 [0.6729, 0.7785, 0.8617, 0.9190, 0.9502, 0.9730],
                 [0.6794, 0.7878, 0.8680, 0.9229, 0.9539, 0.9765],
                [0.6751, 0.7829, 0.8650, 0.9186, 0.9548, 0.9752],
                [0.6749, 0.7755, 0.8609, 0.9175, 0.9519, 0.9740]]) # Data Augmentation


stat_res = []


topk_M_A_df = pd.DataFrame.from_records(top_k_A)
topk_M_PN_df = pd.DataFrame.from_records(top_k_PN)
topk_B_A_df = pd.DataFrame.from_records(binomial_A)
topk_B_PN_df = pd.DataFrame.from_records(binomial_PN)
topk_KL_df = pd.DataFrame.from_records(top_k_KL)




dict={"topk_M_A":topk_M_A_df,
      "topk_M_PN": topk_M_PN_df,
      "topk_B_A":topk_B_A_df,
      "topk_B_PN": topk_B_PN_df,
      "topk_KL_df": topk_KL_df}

colname = ["topk_B_A", "topk_B_PN", "topk_M_A", "topk_KL_df", "topk_M_PN"]

mean_list = []
std_list =[]

#print(dict["topk_M_PN"])
#print(list(dict['topk_M_A'].mean()))

for j in np.arange(len(colname)):
    mean_list.append(dict[colname[j]].mean().tolist())
    std_list.append(dict[colname[j]].std().tolist())

mean_list.append([0.657, 0.77, 0.863, 0.913, 0.948,0.97])
std_list.append([0, 0, 0, 0, 0, 0])
np_mean = np.array(mean_list, dtype=np.float64)
np_std = np.array(std_list, dtype=np.float64)
print("np_mean:", np_mean)
print("np_array:", np_std)

#print(np_mean.shape())
x = np.arange(4)

y = [0.1, 0.2, 0.3, 0.4, 0.5,0.6]


print(np_mean.dtype)
print(type(np_mean))
color = ["orange", "blue","darkred", "green", "purple", "gray"]
label = [r"DRO-TopK$_B$", r"DRO-TopK-PN$_B$", r"DRO-TopK-$_M$", r'DRO-KL$_M$',r"DRO-TopK-PN$_M$", "MS"]
x_0=x
width=0.1
for j in np.arange(6):
    x = x + width
    if j <= 4:
        plt.bar(x,np_mean[j,(0,1,3,5)], width, yerr=np_std[j, (0,1,3,5)], color = color[j], ecolor = 'red', capsize = 2, label = label[j] )

    else:
        plt.bar(x,np_mean[j,(0,1,3,5)], width, color = color[j],  label = label[j] )


#plt.axhline(0.657, xmin=x_0[0], xmax=x[0]+1, linestyle='--')
#plt.axhline(0.77, xmin=x_0[1], xmax=x[1]+1, linestyle='--')
#plt.axhline(0.913, xmin=x_0[2], xmax=x[2]+1, linestyle='--')

print(x)
print(x_0)
#plt.xticks((x+x_0+0.1)/2, [r"DRO-TopK$_B$", r"DRO-TopK-PN$_B$", r"DRO-TopK-A$_M$", r'DRO-KL$_M$',r"DRO-TopK-PN$_M$"])

plt.xticks((x+x_0+0.1)/2, [r"Recall$@1$",r"Recall$@2$", r"Recall$@8$", r"Recall$@32$"])
plt.ylim(0.6, 1)
plt.yticks(fontname="Times New Roman", fontsize=13)
plt.xticks(fontname="Times New Roman", fontsize=11)
plt.ylabel(r"Mean and Std", fontname="Times New Roman", fontsize=17)
plt.xlabel(r"Recall$@k$", fontname="Times New Roman", fontsize=17)

plt.legend(loc = "best", fontsize = 13)
plt.title(r"Mean and Std of Recall$@k$",fontname="Times New Roman", fontweight="bold",fontsize=17)
plt.savefig("/Users/qiqi/Documents/File/a-research/2020-ICLR/Experiment Results/mean_and_std-bar.eps")

plt.show()


# plt.errorbar(x,np_mean[:,0], np_std[:, 0], color = plt.cm.Reds(0.8), solid_capstyle='projecting', linewidth=2, capsize=3, label = r"Recall$@1$")
# plt.axhline(0.657, color = plt.cm.Reds(0.8), linestyle='--')
# plt.errorbar(x,np_mean[:,1], np_std[:, 1], color = plt.cm.Reds(0.7),solid_capstyle='projecting', linewidth=2, capsize=3, label = r"Recall$@2$")
# plt.axhline(0.77, color = plt.cm.Reds(0.7),linestyle='--')
# #plt.errorbar(x,np_mean[:,2], np_std[:, 2], color = plt.cm.Reds(0.6), solid_capstyle='projecting', linewidth=2, capsize=3, label = r"Recall$@4$")
# #plt.axhline(0.863, color = plt.cm.Reds(0.6), linestyle='--')
# plt.errorbar(x,np_mean[:,3], np_std[:, 3], color = plt.cm.Reds(0.6), solid_capstyle='projecting', linewidth=2, capsize=3, label = r"Recall$@8$")
# plt.axhline(0.913, color = plt.cm.Reds(0.6), linestyle='--')
# #plt.errorbar(x,np_mean[:,4], np_std[:, 4], color = plt.cm.Reds(0.4), solid_capstyle='projecting', linewidth=2, capsize=3, label = r"Recall$@16$")
# #plt.axhline(0.948, color = plt.cm.Reds(0.4),linestyle='--')
# plt.errorbar(x,np_mean[:,5], np_std[:, 5], color = plt.cm.Reds(0.5), solid_capstyle='projecting', linewidth=2, capsize=3, label = r"Recall$@32$")
# plt.axhline(0.97, color = plt.cm.Reds(0.5),linestyle='--')
# plt.xticks(x, [r"DRO-TopK$_B$", r"DRO-TopK-PN$_B$", r"DRO-TopK-A$_M$", r'DRO-KL$_M$',r"DRO-TopK-PN$_M$"])
# #.ylim(0.65,0.68)
# plt.legend(loc = (0.75,0.5),fontsize = 10)
# plt.yticks(fontname="Times New Roman", fontsize=13)
# plt.xticks(fontname="Times New Roman", fontsize=11)
# plt.ylabel(r"Recall$@$k", fontname="Times New Roman", fontsize=17)
# plt.title(r"Mean and Std of Recall$@k$",fontname="Times New Roman", fontweight="bold",fontsize=17)
# plt.savefig("/Users/qiqi/Documents/File/a-research/2020-ICLR/Experiment Results/mean_and_std.eps")
# plt.show()

#print(list(dict["topk_B_A"].mean()), list(dict["topk_B_A"].var()))

#print("binomial A.mean")