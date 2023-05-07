#pip install seaborn
import seaborn as sb
import matplotlib.pyplot as plt

iris_DataSet = sb.load_dataset("iris")

# print(iris_DataSet.head())

sb.set()
sb.pairplot(iris_DataSet, hue="species",height=2)
plt.show()