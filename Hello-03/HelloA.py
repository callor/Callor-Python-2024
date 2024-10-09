import matplotlib
import matplotlib.pyplot as plt
print(matplotlib.__version__)
plt.plot([10,20,30,40,50])
plt.show()

import numpy as np
 
x = np.linspace(0,15,150)
y = np.sin(x)
 
plt.plot(x,y)
plt.show()