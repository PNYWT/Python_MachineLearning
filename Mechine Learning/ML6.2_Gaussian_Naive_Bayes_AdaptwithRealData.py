#ทำนายรายได้ประชากรด้วย GaussianNB จากข้อมูลที่เก็บจริง ไม่ได้โหลดมากจาก sklearn
"""
ดู Attribute เช่น การศึกษา เพศ การแต่งงาน จากนั้นประเมินหา income ว่าจะได้มากกว่าหรือน้อยกว่า 50k
Attribute คือ x
มากกว่าหรือน้อยกว่า 50k เป็น Class คือ y_Test ผลเฉลย
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# func Clean Data เพราะมีข้อมูลบางส่วนที่เป็นไม่ระบุค่า หรือตัวเลขแปลกๆ
# ในที่นี้เราจะจัดทำเข้ารหัสข้อความให้เป็นตัวเลขทั้งหมด
def cleanData(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object): #วนหาค่า dataset ในแต่ละ idx ว่าเป็น type object หรือไม่
            lbEncoder = LabelEncoder()
            dataset[column] = lbEncoder.fit_transform(dataset[column]) #เข้ารหัสข้อความให้เป็นตัวเลข
    return dataset

# func split แบ่งเพื่อเลือก Attribute(เลือก age -> country) และ class(เลือก income) แยกเก็บลงในตัวแปร
def split_Attribute_Class(dataset,columnName): 
    attribute_Select = dataset.drop(columnName, axis=1) #Attribute(เลือก age -> country)
    class_Select = dataset[columnName].copy() #class(เลือก income)
    return attribute_Select, class_Select

#local
ROOT_DIR = os.path.abspath(os.curdir)
path = "{}\{}".format(ROOT_DIR,"Mechine Learning\FileData\/adult_census_income.csv")
dataset = pd.read_csv(path)
#หลังจาก Clean Data แล้ว income <= 50k คือ 0 และ >=50k คือ 1
dataset = cleanData(dataset)
print(dataset.head())

# Split data จาก Data ทั้งหมดแยกเป็น Training-Test
data_Training, data_Test = train_test_split(dataset, train_size=0.7, test_size=0.3)

#เลือก Attribute(เลือก age -> country) และเลือก class(เลือก income) เก็บในตัวแปร สำหรับเข้า Traning model
x_Training, y_Training = split_Attribute_Class(data_Training, "income")
#เลือก Attribute(เลือก age -> country) และเลือก class(เลือก income) เก็บในตัวแปร สำหรับเข้า Test model
x_Test, y_Test = split_Attribute_Class(data_Test, "income")

#Select model
model = GaussianNB()
model.fit(x_Training, y_Training)

#prediction
y_Predic = model.predict(x_Test)

# วัดประสิทธิภาพของ model เปรียบเทียบกับผลเฉลย y_Test โดยเราจะดู Accuracy Score หรือ classification_report
print("Accuracy -> {0:.2f} %".format(accuracy_score(y_Test, y_Predic)*100))
# print(classification_report(y_Test, y_Predic))
print(pd.crosstab(y_Test, y_Predic, rownames=["Real"], colnames=["Predict"], margins=True))
"""
Accuracy -> 79.38 % ยังถือว่าไม่ผ่านตามกำหนด
"""