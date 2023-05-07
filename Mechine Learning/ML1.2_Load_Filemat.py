from scipy.io import loadmat #use to load .mat
import matplotlib.pyplot as plt #create graph
import os
ROOT_DIR = os.path.abspath(os.curdir)
path = "{}\{}".format(ROOT_DIR,"Mechine Learning\FileData\mnist-original.mat")
mnist_Raw = loadmat(path)
# print("minst_raw -> {}".format(mnist_Raw))

mnist_New = {
    "data":mnist_Raw["data"].T, # (28x28 = 784, record 70000) ->  .T is ( record 70000 , 28x28 = 784)
    "target":mnist_Raw["label"][0]
}
print("mnist_New data shape -> {}".format(mnist_New["data"].shape)) #mnist_New data -> (28x28 = 784, record 70000)

data_Record = mnist_New["data"] #(784, record 70000)
# print("data_Record -> {}".format(data_Record.shape))
data_Label = mnist_New["target"] # [0. 0. 0. ... 9. 9. 9.]
# print("data_Label -> {}".format(data_Label))

plot_Data_record = data_Record[30000]
plot_Data_record_ReshapeImg = plot_Data_record.reshape(28,28)
plt.imshow(plot_Data_record_ReshapeImg, cmap=plt.cm.binary, interpolation="nearest")
print("Label is -> {}".format(data_Label[30000]))
plt.show()
