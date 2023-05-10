from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

#load data set
iris_Data = load_iris()
"""
ดูโครงสร้าง data ว่ามี อะไรบ้าง
df = pd.DataFrame(data=iris_Data.data, columns=iris_Data.feature_names)
df['target'] = iris_Data.target
print(df.head())
"""
x_Iris = iris_Data.data  #Attribute เช่น ความกว้างใบ ความยาวใบ เป็นต้น
y_Iris = iris_Data.target #ชื่อสายพันธ์



#split data Training-test
x_Training, x_Test, y_Training, y_Test = train_test_split(x_Iris, y_Iris, train_size=0.7, test_size=0.3)

#select model GaussianNB
model = GaussianNB()

#Training model
model.fit(x_Training, y_Training)

# หลัง Training แล้วก็จะ ทดสอบ model โดยการหา y_Predic
# จากนั้นนำ y_Predic มาเทียบกับ y_Test ซึ่งเป็นผลเฉลย ว่ามีค่าใกล้เคียงกันหรือไม่
y_Predic = model.predict(x_Test)

# วัดประสิทธิภาพของ model โดยเราจะดู Accuracy Score หรือ classification_report
print("Accuracy -> {0:.2f} %".format(accuracy_score(y_Test, y_Predic)*100))
# print(classification_report(y_Test, y_Predic))
print(pd.crosstab(y_Test, y_Predic, rownames=["Real"], colnames=["Predict"], margins=True))
"""
มีการทำนายถูกถึง 95.56 % ถือว่าเป็นที่ยอมรับ สามารถนำ model ตัวนี้ไปใช้งานกับข้อมูลจริงได้
สรุปผลได้ดังนี้
Versicolor , Setosa , Virginica
เป็น ทายเป็น Versicolor ทั้งหมดจาก 17 ข้อมูล
เป็น ทายเป็น Setosa จำนวน 12 ข้อมูล และ อาจจะเป็น Virginica 1 ข้อมูล จากทัั้งหมด 13 ข้อมูล
เป็น ทายเป็น Virginica จำนวน 14 ข้อมูล และ อาจจะเป็น Setosa 1 ข้อมูล จากทัั้งหมด 15 ข้อมูล
"""