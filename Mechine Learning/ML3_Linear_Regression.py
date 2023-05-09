#https://kongruksiam.medium.com/สรุป-machine-learning-ep-3-การวิเคราะห์การถดถอยเชิงเส้น-linear-regression-891260e4a957
"""
Linear regression เป็น การเรียนรู้แบบมีผู้สอน (Supervised Machine Learning Algorithms) ในข้อมูลมีชุดเฉลยอยู่แล้ว
Linear regression สมการ -> y = ax+b
x = ตัวแปรที่ทราบค่า | ตัวประมาณการ(Predictor)
y = ตัวแปรที่เราไม่ทราบค่า | ตัวตอบสนอง (Response) <- ส่วนใหญ่แล้วเราต้องการหาตัวนี้
a = ความชันของเส้นตรง
b = ระยะตัดแกน y
"""

#pip install numpy
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression #นำเข้า LinearRegression Model

#Ex 1 แบบ plot (การจำลองข้อมูล)
"""
x = np.linspace(-5,5,100) #สร้างข้อมูลตั้งแต่ -5 ถึง 5 จำนวน 100 records (record ถ้าไม่ใส่จะ Run มาให้ 50 records)
# กำหนดให้ a = 2 , b = 1
y = (2*x) + 1

plt.plot(x,y,"-b",label="y = 2x + 1")
plt.title("Linear regression Ex1 -> y = 2x + 1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper right")
plt.grid()
plt.show()
"""

#Ex2 แบบ Scatter การกระจายข้อมูล (การจำลองข้อมูล)
"""
randome_Range = np.random
x = randome_Range.rand(50) #สุ่มค่าบวก
a = 2
b = randome_Range.randn(50) #สุ่มทั้งค่าบวก และ ลบ
y = a*x + b
plt.scatter(x,y)
plt.title("Linear regression Ex2 -> y = a*x + b")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""

#Ex3 Step การทำงานเบื้องต้นดังนี้ --->
# 1.เลือก model ที่จะใช้ทำการ Training และ Test
# 2.นำเข้าข้อมูล Training และทำการทดสอบ model
# 3.นำเข้าข้อมูล test และทำการทดสอบ model
# 4.Analysis result model

# 1.เลือก model ที่จะใช้ทำการ Training และ Test
model_LinearRe = LinearRegression()

# 2.นำเข้าข้อมูล Training และทำการทดสอบ model 
randome_Range = np.random
training_X = randome_Range.rand(50)*10
a = 2
b = randome_Range.randn(50)
training_Y = a*training_X + b

#2.1 แปลง ข้อมูล Training จาก arr1D -> 2D
# print("training_X Array 1D -> {}".format(training_X))
convert_training_X_to_Array2D = training_X.reshape(-1,1)
# print("convert_training_X_to_Array2D 2D -> {}".format(convert_training_X_to_Array2D))

#2.2 Trainging model
model_LinearRe.fit(convert_training_X_to_Array2D,training_Y)
"""
Coefficient (ค่าความสัมพันธ์)
    ค่าสัมประสิทธิ์แสดงการตัดสินใจ คือ ตัวเลขที่บอกความสัมพันธ์ของสองตัวแปร หรือ ค่าที่แสดงว่าตัวแปร x มีอิทธิพลต่อตัวแปร y มากน้อยเพียงใดโดยมี ขอบเขตในช่วง -1 ถึง 1
#Coefficient
print(model_LinearRe.coef_)
    
Intercept คือ ค่าที่บ่งบอกจุดตัดแกน
#Intercept
print(model_LinearRe.intercept_)

R-Square คือ ค่าความผันแปรของตัวแปร y มีค่าอยู่ระหว่าง 0% — 100%
    -> 0% หมายถึง ผลลัพธ์ที่ได้มานั้นไม่สามารถอธิบายความผันแปรของค่าตัวแปร y ต่างที่กระจายรอบค่าเฉลี่ยได้เลย
    -> 100% แสดงให้เห็นว่าผลลัพธ์ที่ได้มานั้นสามารถอธิบายความผันแปรของค่าตัวแปร y ต่างที่กระจายรอบค่าเฉลี่ยได้เป็นอย่างดี
#R-Square ใช้วัดผลว่า model แม่นยำมากแค่ไหน
print(model_LinearRe.score(convert_X_to_Array2D,y)*100)
"""

# 3.นำเข้าข้อมูล test และทำการทดสอบ model
test_X = np.linspace(-1,11) #สร้างข้อมูลตั้งแต่ -1 ถึง 1 จำนวน 50 records
convert_test_X_to_Array2D  = test_X.reshape(-1,1)
# print(convert_test_X_to_Array2D.shape)
# predict หา y
test_y_predict = model_LinearRe.predict(convert_test_X_to_Array2D)

# 4.Analysis result model
plt.title("Linear Regression")
plt.scatter(training_X,training_Y)
plt.plot(test_X,test_y_predict)
plt.show()