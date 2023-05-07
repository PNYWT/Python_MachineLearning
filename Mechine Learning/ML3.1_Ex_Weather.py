import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Online
"""
dataset = pd.read_csv("https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv")
print(dataset.shape)
"""

#local
ROOT_DIR = os.path.abspath(os.curdir)
path = "{}\{}".format(ROOT_DIR,"Mechine Learning\FileData\Weather.csv")
dataset = pd.read_csv(path)
# print(dataset.shape)
# print(dataset.describe()) #แสดง รายละเอียดของ file แบบ สรุป
# ใน Ex นี้ เราจะใช้ col MaxTemp และ MinTemp


# แปลง Array 1D -> 2D
x_DataSet = dataset["MinTemp"].values.reshape(-1,1)
y_DataSet = dataset["MaxTemp"].values.reshape(-1,1)

#split data 70,30
training_X,test_X,training_Y,test_Y = train_test_split(x_DataSet, y_DataSet, train_size=0.7, test_size=0.3)

#select model
model = LinearRegression()

#Training model
model.fit(training_X, training_Y)
# ตรวจสอบความแม่นยำของ model หลัง training, test model
print("Train set accuracy = " + str(model.score(training_X, training_Y)))
print("Test set accuracy = " + str(model.score(test_X, test_Y)))

# test model ทำนายหา predict_Y
# การจะเอามาทำนาย Y นั้น model ต้องมีความแม่นยำพอสมควรแล้วจึงจะมาทำขั้นตอนนี้
predict_Y = model.predict(test_X)
"""
# ตรวจสอบ predict_Y เป็นอย่างไร ถ้าผ่ากลางกลุ่มข้อมูลตัวอย่างแบบพอดีๆ ก็อาจจะยืนยันได้ว่า Model มีความแม่นยำพอสมควร
plt.scatter(test_X, test_Y)
plt.xlabel("test_X")
plt.ylabel("test_Y")
plt.plot(test_X, predict_Y, color="red", linewidth = 2)
plt.show()
"""

# compare predict data vs Real data (data training)
"""
การวัดประสิทธิภาพ
    test_Y คือ ข้อมูลจริงที่ใช้ทดสอบการทำนายผลโมเดล
    predict_Y คือผลการทำนายผลลัพธ์จากโมเดล

Loss Function (ยิ่งน้อยยิ่งดี แต่อย่าง Bias)
    คือ การคำนวน Error ว่า predict_Y ที่โมเดลทำนายออกมา ต่างจาก test_Y อยู่เท่าไร 
    แล้วหาค่าเฉลี่ย เพื่อที่จะนำมาหา Gradient ของ Loss แล้วใช้อัลกอริทึม Gradient Descent เพื่อให้ Loss น้อยลงในการเทรนรอบถัดไป

Loss Function ที่นิยมใช้ในงาน Regression ได้แก่
    1. Mean Absolute Error (MAE)
        การคำนวน Error ว่า predict_Y ต่างจาก test_Y อยู่เท่าไร ด้วยการนำมาลบกันตรง ๆ แล้วหาค่าเฉลี่ย 
        โดยไม่สนใจเครื่องหมาย (Absolute) เพื่อหาขนาดของ Error โดยไม่สนใจทิศทาง
    2. Mean Squared Error (MSE)
        การคำนวน Error ว่า predict_Y ต่างจาก test_Y อยู่เท่าไร ด้วยการนำมาลบกัน แล้วยกกำลังสอง (Squared) 
        เพื่อไม่ต้องสนใจค่าติดลบ (ถ้ามี) แล้วหาค่าเฉลี่ย
    3. Root Mean Squared Error (RMSE)
        นำ MSE มาหา Squared Root
"""
# แปลง Array 2D -> 1D
test_Y_1D = test_Y.flatten()
predict_Y_1D = predict_Y.flatten()
# Real data -> ข้อมูลที่ Training + test มาแล้ว
# Predicted Data -> ค่าที่เราทำนายมา หลังจากผ่านการ Training + test มาแล้ว
df = pd.DataFrame({"Real data":test_Y_1D, "Predicted Data":predict_Y_1D})
# print(df.head())

"""
# การดูแบบนี้ด้วยตาอาจจะบอกอะไรไม่ได้มาก ดังนั้นเราต้องเช็คด้วยค่าทางสถิติต่อ
df_Row = df.head(20)
df_Row.plot(kind="bar", figsize=(10,10))
plt.show()
"""

# ตรวจสอบด้วยค่าทางสถิติ
MAE_result = metrics.mean_absolute_error(test_Y, predict_Y)
MSE_result = metrics.mean_squared_error(test_Y, predict_Y)
RMSE_result = np.sqrt(metrics.mean_squared_error(test_Y, predict_Y))
"""
ค่า MSE, MAE อยู่ในช่วง 0 — Infinity เหมือนกัน ยิ่งน้อยคือยิ่งดีถ้าเป็น 0 คือ ไม่ มีError เลย 
ดังนั้นถ้าค่าเท่ากับ 0 แปลว่าโมเดลทำนายค่า test_Y ได้ถูกต้อง 100% 
แต่ในทางปฏิบัติโอกาสที่จะเทรนโมเดลได้ loss = 0 เป็นไปได้ยากมาก
"""
"""
การเปรียบเทียบก็คือ เรามักจะนำค่า MAE, MSE หรือ RMSE ค่าใดค่าหนึ่งจากโมเดลเก่าและโมเดลใหม่มาวางคู่กันแล้วบอกว่า 
โมเดลไหนดีกว่ากัน ซึ่งเป็นสิ่งที่ควรระวังอย่างยิ่ง เนื่องจากสูตรคำนวนค่า Error นั้นจะมีค่า n หรือจำนวนของข้อมูลมาเกี่ยวข้องด้วย
cr.https://medium.com/c-g-datacommunity/mse-rmse-mae-เลือกใช้ยังไงดีมาลองดูที่ความหมาย-17b37b0b14b3
"""
print("MAE ค่าเฉลี่ยของค่า Error นั้นมีค่าเท่ากับ = {}".format(MAE_result))
print("MSE = {}".format(MSE_result))
print("RMSE = {}".format(RMSE_result))

#แสดงค่าความแม่นยำด้วย R-Square หากมีค่าเป็น 1 แสดงว่าแม่นยำที่สุด
Rsqur_score = metrics.r2_score(test_Y,predict_Y)
print("Score = {0:.4f}, คิดเป็น {1:.2f} %".format(Rsqur_score, Rsqur_score*100))
print("ถ้าต่ำกว่า 80% ในเชิงการวิเคราะห์ มองว่า model นี้ยังไม่แม่นยำเท่าที่ควร จึงจะยังไม่นำไปใช้")