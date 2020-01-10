import numpy as np
import math
import matplotlib.pyplot as plt
def pdf_distance(d, n):
    y_n = d**(n-2) * (1-0.25*d**2)**((n-3)/2)

    return y_n

d = np.arange(0, 2,0.01)


pdf_512 = pdf_distance(d, 512.0)
pdf_1024 = pdf_distance(d, 1024.0)
plt.figure()
plt.plot(d,1/pdf_512, label="DIM 512")
plt.plot(d,1/pdf_1024, label='DIM 1024')
plt.legend(loc = 'best')
plt.xlabel("d")
plt.ylabel("q(d)")
plt.ylim(0,100)
plt.show()

print(d)