import pylab #pip install matplotlib
from sklearn import datasets

digits_Datasets = datasets.load_digits()

"""
Key of iris_Datasets
    dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
    target_names -> เก็บตัวเลขลายมือ 0-9
    images -> ภาพตัวเลข 8x8 px
"""
# print(digits_Datasets.target_names[0])
print("Number -> {}".format(digits_Datasets.target_names[1]))
pylab.imshow(digits_Datasets.images[1], cmap=pylab.cm.gray_r)
pylab.show()