import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# func แสดงผลภาพตัวเลข predict_Number ที่ใส่เข้ามา
def displayImage(x):
    plt.imshow(x.reshape(28,28), cmap= plt.cm.binary, interpolation="nearest")
    plt.show()

# func แสดงผลเฉลยต predict_Number อยู่ใน class ไหน แล้วตรงกับ ความเป็นจริงของข้อมูลหรือไม่
def displayPredict(classifier,realClass_Y,x_set):
    print("Prediction -> {}".format(classifier.predict([x_set])[0]))
    print("Real class is -> {}".format(realClass_Y))
   
# func แสดงผล ConfusionMatrix ต้อง import itertools
def displayConfusionMatrix(con_Matrix,cmap=plt.cm.GnBu):
    classes=["Other Number","Number 5"]
    plt.imshow(con_Matrix,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes))
    plt.xticks(trick_marks,classes)
    plt.yticks(trick_marks,classes)
    thresh=con_Matrix.max()/2
    for i , j in itertools.product(range(con_Matrix.shape[0]),range(con_Matrix.shape[1])):
        plt.text(j,i,format(con_Matrix[i,j],'d'),
        horizontalalignment='center',
        color='white' if con_Matrix[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

ROOT_DIR = os.path.abspath(os.curdir)
path = "{}\{}".format(ROOT_DIR,"Mechine Learning\FileData\mnist-original.mat")
mnist_Raw = loadmat(path)
mnist= {
    "data":mnist_Raw["data"].T,
    "target":mnist_Raw["label"][0]
}
# print("data -> {}".format(mnist["data"].shape))
# print("target -> {}".format(mnist["target"].shape))
x_Data = (mnist["data"])
y_Data = (mnist["target"])

# slipt data to training set and test set (แบ่งข้อมูล)
# training set  -> 1 - 60000
# test set -> 60001 - 70000
# เรียงจาก class 0 -> 9
x_Training , x_Test, y_Training, y_Test = x_Data[:60000], x_Data[60000:], y_Data[:60000], y_Data[60000:]
"""
print("x_Training -> {}".format(x_Training.shape))
print("x_Test -> {}".format(x_Test.shape))
print("y_Training -> {}".format(y_Training.shape))
print("y_Test -> {}".format(y_Test.shape))
"""

# จำแนกข้อมูลเป็น 2 กลุ่ม Binary Classification แล้วเอาชุดนี้ไปใช้ในการ Training-Test and Predict
y_Training_Classifier = (y_Training == 5) # โดยการจำแนก class เป็นเลข 5 True, class != 5 False
y_Test_Classifier = (y_Test == 5) # โดยการจำแนก class เป็นเลข 5 True, class != 5 False

# print(y_Training_Classifier.shape, y_Training_Classifier)
# print(y_Test_Classifier.shape, y_Test_Classifier)

# select model
sgd_Classifier = SGDClassifier()
# Training Data
sgd_Classifier.fit(x_Training, y_Training_Classifier)

"""
# ใช้ดูตำแหน่งที่ต้องการทราบว่าอยู่ใน class ไหน == 5 หรือ != 5
predict_Number = 4000
displayPredict(sgd_Classifier, y_Test_Classifier[predict_Number], x_Test[predict_Number])
displayImage(x_Test[predict_Number])
"""

"""
#การวัดประสิทธิภาพ ด้วย Cross-validation Test
Cross-validation Test การทดสอบประสิทธิภาพของโมเดลด้วยวิธี Cross-validation นี้จะทําการแบ่งข้อมูลออกเป็นหลายๆส่วน (k) 
เช่นกำหนดให้ k-fold=3 แสดงว่ามีการแบ่งข้อมูลออกเป็น 3 ส่วน หรือการทดลอง 3 ครั้งโดยผลลัพธ์ที่ได้จะบอก 
ผลการทดลองครั้งที่ 1 , 2 และ 3 ตามลำดับ

ใน Machine Learning หลังจากที่เราเราเทรนโมเดลใช้ในงานต่าง ๆจะมีการคำนวน Metrics เพื่อแสดงผลให้ทราบว่าโมเดลนั้น ๆ ทำงานได้ดีแค่ไหน
ซึ่งจะยกตัวอย่างโดยใช้ Accuracy, Precision, Recall และ F1 Score
ถ้าค่าความแม่นยำมากกว่า 90% ขึ้นไปสำหรับข้อมูลขนาดเล็กๆก็ถือว่าโมเดลมีประสิทธิภาพดี 
แต่ในบางกรณีข้อมูลของเรามีจำนวนเยอะมากความแม่นยำก็อาจจะลดลงบ้างถือเป็นเรื่องปกติ

from sklearn.model_selection import cross_val_score
# k-fold คือ parameter cv
# ใช้สำหรับดู score Training รวมๆ แต่ถ้าจะดูผลลัพธ์เป็นรอบๆ ต้องใช้ confusion_matrix
score_of_Model = cross_val_score(sgd_Classifier,x_Training,y_Training_Classifier, cv=3, scoring="accuracy")
print("score_of_Model -> {}".format(score_of_Model))
"""

# ดูผลลัพธ์ real data vs predict
# sgd_Classifier training-test มาแล้วถึง นำมา cross_val_predict เพื่อทำนาย Y ใช้ข้อมูลชุด Training set
predict_Y_Training = cross_val_predict(sgd_Classifier,x_Training,y_Training_Classifier, cv=3)
con_Matrix = confusion_matrix(y_Training_Classifier, predict_Y_Training) # จะได้เป็น Array 2D
"""
cv = 3
print(con_Matrix)
[[48557  6022]
 [ 1182  4239]]

 อ่านผล
 ตัวเลขที่ไม่ใช่เลขที่กำหนด ทำนายถูก = 48557, อาจจะทำนายผิด 6022
 เป็นเลขที่กำหนด ทำนายถูก = 4239, อาจจะทำนายผิด 1182

จากที่ลองทดสอบจำนวนรอบ cv =  3,5,10 ตามลำดับ
พบว่า cv= 5 จำนวนการทำนาย Other Number and Number 5 ถูกน้อยกว่า cv = 3 และ 10

plt.figure()
displayConfusionMatrix(con_Matrix)
"""

# สร้าง Classification Report
"""
ใน Machine Learning หลังจากที่เราเราเทรนโมเดลใช้ในงานต่าง ๆจะมีการคำนวน Metrics 
เพื่อแสดงผลให้ทราบว่าโมเดลนั้น ๆ ทำงานได้ดีแค่ไหนซึ่งจะยกตัวอย่างโดยใช้ Accuracy, Precision, Recall และ F1 Score
ถ้าค่าความแม่นยำมากกว่า 90% ขึ้นไปสำหรับข้อมูลขนาดเล็กๆก็ถือว่าโมเดลมีประสิทธิภาพดี 
แต่ในบางกรณีข้อมูลของเรามีจำนวนเยอะมากความแม่นยำก็อาจจะลดลงบ้างถือเป็นเรื่องปกติ

ในส่วนของ Accuracy, Precision, Recall และ F1 Score จะอาศัยข้อมูลจาก Confusion Matrix มาคำนวณค่า
https://kongruksiam.medium.com/สรุป-machine-learning-ep-4-ตัวจำแนกแบบไบรารี่-binary-classifier-6ebc8e1a5e61
ดูตาราง คำจำกัดความ
True Positive (TP) - ทายว่าถูกต้อง แล้วตอบตรงตามที่ทายไว้ (ทายถูกอยู่ในกลุ่มที่เรากำหนดจริง (ถูก))
True Negative (TN) - ทายว่าไม่ถูกต้อง แล้วตอบตรงตามที่ทายไว้ (ทายว่าไม่ได้อยู่ในกลุ่มที่เรากำหนดจริง (ถูก))
False Positive (FP) - ทายว่าถูกต้อง แต่คำตอบคือไม่ถูกต้อง (ทายถูก แต่กลุ่มที่กำหนดไม่ถูก)
False Negative (FN) - ทายว่าไม่ถูกต้อง แต่คำตอบคือถูกต้อง (ทายผิด แต่บอกถูกกลุ่ม)

True Positive (TP) || False Positive (FP) 
___________________||____________________
False Negative(FN) || True Negative (TN) 
"""
# predict หา Y โดยใช้ข้อมูลชุด test set
predict_Y_Test = sgd_Classifier.predict(x_Test)

className = ["Other Number", "Number 5"]
# ดูผลลัพธ์ real data vs predict
print(classification_report(y_Test_Classifier,predict_Y_Test, target_names=className))
print("Accuracy -> {0:.2f} %".format(accuracy_score(y_Test_Classifier,predict_Y_Test)*100))