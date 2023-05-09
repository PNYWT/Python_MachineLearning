#https://kongruksiam.medium.com/สรุป-machine-learning-ep-4-เพื่อนบ้านใกล้ที่สุด-k-nearest-neighbors-787665f7c09d
"""
เป็น การเรียนรู้แบบมีผู้สอน (Supervised Machine Learning Algorithms) ในข้อมูลมีชุดเฉลยอยู่แล้ว
K-Nearest Neighbors (K-NN) เป็นวิธีการแบ่งคลาสสำหรับใช้จัดหมวดหมู่ข้อมูล (Classification)
ใช้หลักการเปรียบเทียบข้อมูลที่สนใจกับข้อมูลอื่นว่ามีความคล้ายคลึงมากน้อยเพียงใด หากข้อมูลที่กำลังสนใจนั้นอยู่ใกล้ข้อมูลใดมากที่สุด 
ระบบจะให้คำตอบเป็นเหมือนคำตอบของข้อมูลที่อยู่ใกล้ที่สุดนั้นลักษณะการทำงานแบบไม่ได้ใช้ข้อมูลชุดเรียนรู้ (training data) 
ในการสร้างแบบจำลองแต่จะใช้ข้อมูลนี้มาเป็นตัวแบบจำลองเลย
    - เทคนิคนี้จะตัดสินใจว่าข้อมูลมีความคล้ายคลึงหรือใกล้เคียงกับคลาสใด โดยการตรวจจสอบข้อมูลบางจำนวน (K)
    - เหมาะสำหรับข้อมูลแบบตัวเลขเพื่อหาวิธีการวัดระยะห่างของแต่ละ Attribute ในข้อมูลให้ได้

ตัวอย่าง เช่น ความเสี่ยงในการเกิดโรคเบาหวาน
โดยหาความสัมพันธ์ของข้อมูลเช่น ระยะเวลาการตั้งครรภ์, Glucose, ความดันในเลือด, BMI, อายุ เป็นต้น
เราจะเอาข้อมูลที่ได้กล่าวไปของแต่ละคนมาทำการ Training-test model จากนั้นจะได้ค่าความสัมพันธ์มา
เมื่อได้กค่าความสัมพันธ์แล้วก็เอาไปเทียบกับคำตอบที่เรามีอยู่แล้วว่าตรงกัน/ใกล้เคียงกันหรือไม่

ข้อดีของ KNN
    1.หากเงื่อนไขในการตัดสินใจมีความซับซ้อนวิธีนี้สามารถนำไปสร้างโมเดลที่มีประสิทธิภาพได้

ข้อเสียของ KNN
    1.ใช้เวลาคำนวณนาน
    2.ถ้า Attribute มีจำนวนมากจะเกิดข้อผิดพลาดในการคำนวณค่า
    3.คำนวณค่าได้เฉพาะข้อมูลประเภท Nominal เช่น ข้อมูลเพศชาย-หญิง อาชีพ
"""
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ดึง Data จาก Local path ที่เก็บไว้
ROOT_DIR = os.path.abspath(os.curdir)
path = "{}\{}".format(ROOT_DIR,"Mechine Learning\FileData\diabetes.csv")
df = pd.read_csv(path)

#แสดงข้อมูล 5 แถวแรก
"""
print(df.head())
"""

#แสดงโครงสร้าง  df -> 768 rows, 9 columns
"""
print(df.shape)
"""

#ใช้ 8 columns ในการ Training-Test เก็บไว้ใน x_Data, ใน column ที่ 9 เราจะ drop ออกมาเก็บไว้ใน y_Data
# y_Data ค่อยเอาไปเทียบทีหลังว่า model คำนวณออกมาตรงกับผลเฉลยไหม
# x_Data, y_Data จะเป็น array 2D
x_Data = df.drop("Outcome",axis=1).values #drop Outcome ออก เก็บไว้แค่ Pregnancies...Age
y_Data =df["Outcome"].values #เก็บผลเฉลยไว้ใช้ Compare กับ ที่ model ทำนาย

#split data
x_Training, x_Test, y_Training, y_Test = train_test_split(x_Data, y_Data, train_size=0.7, test_size=0.3)

#แสดงโครงสร้างหลังแบ่ง Training:Test, 70:30
"""
print(x_Training.shape)
print(x_Test.shape)
"""

# Training model หาค่า K ที่ดีที่สุด โดยใส่ค่า K ลงไปเลย
# คือเราไม่รู้ว่าค่า K ตั้งแต่ 1-10 อันไหนเป็นค่าที่ดีที่สุด เราก็เลยต้องทำแบบนี้
k_Value = np.arange(1,11)
training_Score = np.empty(len(k_Value)) #array empty
test_Score = np.empty(len(k_Value)) #array empty

"""
for score,per_k in enumerate(k_Value) :
    # print("k------>",per_k)
    knn_Model = KNeighborsClassifier(n_neighbors=per_k)
    knn_Model.fit(x_Training, y_Training)
    #วัดประสิทธิภาพค่า K ของ Model แล้วเก็บไว้ดูว่าอันไหน work สุด
    training_Score[score] = knn_Model.score(x_Training, y_Training) #เก็บประสิทธิภาพของการ Training
    test_Score[score] = knn_Model.score(x_Test, y_Test) #เก็บประสิทธิภาพของการ Test

    print("Check test score -> {0:.2f} %".format(test_Score[score]*100))

# Compare k1-10 โดยสร้างกราฟเปรียบเทียบ
plt.title("Compare k 1-10 in Model")
plt.plot(k_Value, test_Score, label="Training Score") #(x -> k_Value,y -> test_Score)
plt.plot(k_Value, training_Score, label="Test Score")
plt.legend()
plt.xlabel("k Value")
plt.ylabel("Score")
plt.show()
#Training Score จะเลือกใช้ค่า K ที่สูงที่สุด และค่อนข้างที่จะคงที่ประมาณ 2-3 ค่า ใช้ดูดูว่า model ของเรากำลังมีปัญหา bias หรือ variance
#Test Score จะเลือกใช้ค่า K ที่สูงที่สุด และค่อนข้างที่จะคงที่ประมาณ 2-3 ค่า นำไปใช้ทดสอบกับข้อมูลจริง ก่อนจะนำไปใช้งานจริงๆ
"""

# เมื่อได้ค่า K test score ที่อยากได้ มาแล้วก็เอามา Train model ในที่นี้ เราจะใช้เลข 6 จากการหาค่า K ครั้งล่าสุด
# Predict ผู้ป่วยโรคเบาหวานด้วย KNN model และ วัดประสิทธิภาพ model
k_Select = 10
knn_Model = KNeighborsClassifier(n_neighbors = k_Select )
knn_Model.fit(x_Training, y_Training)

# Prediction ผู้ป่วยโรคเบาหวานด้วย
y_Predict = knn_Model.predict(x_Test)

# Compare Model ค่า y_Predict เทียบกับ y_Test
print(classification_report(y_Test,y_Predict))
"""
K = 6
Report ผู้ป่วยที่เราทำนายนั้นมีจำนวน 231 คน หรือ 30% จากชุดข้อมูล test set
ผู้ป่วยที่เป็นโรคเบาหวาน 153 คน
ไม่ได้ป่วย 78 คน
Accuracy (ค่าความถูกต้อง) -> 73 % ยังต่ำกว่า 80% ถือว่ายังไม่เป็นที่ยอมรับ
"""

# ดู confusion_matrix เพื่อดูว่าทำไม Prediction ถึงคลาดเคลื่อนจากความเป็นจริง
print("-----------------------")
print(pd.crosstab(y_Test, y_Predict, rownames=["Real"], colnames=["Predict"], margins=True))