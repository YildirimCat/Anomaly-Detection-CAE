import numpy as np
import pandas as pd
import tsia
from matplotlib import pyplot as plt
from pyts.image import GramianAngularField

merged_data = pd.DataFrame()
merged_data = merged_data.append(
    pd.read_csv('./Averaged_Bearing_Dataset_test_2.csv', index_col=0))  # Avoiding unnamed=0 index
print(merged_data.head())

train = merged_data['2004-02-12 10:32:39':'2004-02-15 05:02:39']
test = merged_data[:]

print('Training dataset shape: ', train.shape)
print('Test dataset shape: ', test.shape)

g_train = train.values.reshape(1, -1)
g_test = test.values.reshape(1, -1)

print('Training dataset shape: ', train.shape)
print('Test dataset shape: ', test.shape)

gaf = GramianAngularField(image_size=4, method='s', sample_range=(-1, 1), overlapping=False, flatten=False)

for i in range(0, 400):
    time_slice = test[i:i+1]
    gaf_sample_slice = gaf.fit_transform(X=time_slice)
    plt.imshow(gaf_sample_slice[0])
    plt.savefig("gaf_normal_train_{}".format(i))
    #plt.show()


gaf_train = gaf.transform(X=g_train)
gaf_test = gaf.transform(X=g_test)

plt.imshow(gaf_train[0])
plt.show()
plt.imshow(gaf_test[0])
plt.show()