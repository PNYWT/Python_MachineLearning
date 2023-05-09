#https://kongruksiam.medium.com/สรุป-machine-learning-ep-2-รู้จักกับข้อมูลชุดเรียนรู้และข้อมูลชุดทดสอบ-119a16a901c8
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split #use to split data to training and test model

iris_DataSet = load_iris()
# print("iris_DataSet shape -> {}".format(iris_DataSet.data.shape)) # ( recoad 150, 4 types)

"""
Key of iris_Datasets
    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

    target -> สายพันธุ์ เป็น รหัสตัวเลข
    target_names -> สายพันธุ์ เป็น ชื่อ
    DESCR -> รายละเอียด คำอธิบาย
    feature_names -> ข้อมูล ความกว้าง ยาว สูง  หน่วย cm ของแต่ละสายพันธุ์
    data -> ข้อมูลที่บันทึก ความกว้าง ยาว สูง เป็นตัวเลข ของแต่ละสายพันธุ์ 
"""
#Defualt of lib is 75%:25%
# Training 70% : Test 30%
# training x , test x , training y , test y = train_test_split()
data_Training_X,data_Test_X,data_Training_Y,data_Test_Y = train_test_split(iris_DataSet.data, iris_DataSet.target, random_state=0, train_size=0.7 ,test_size=0.3)

print("data_Training_X -> {}".format(data_Training_X.shape))
print("data_Training_Y -> {}".format(data_Training_Y.shape))
print("data_Test_X -> {}".format(data_Test_X.shape))
print("data_Test_Y -> {}".format(data_Test_Y.shape))

"""
https://www.mindphp.com/บทเรียนออนไลน์/python-tensorflow/8576-how-to-split-dataset.html
https://medium.com/botnoi-classroom/คำแนะนำเมื่อต้องนำ-machine-learning-ไปประยุกต์ใช้งานจริง-d3d779fe14ae
training set ใช้ดู error หรือ bias การปรับค่านั้นนี่โน้น เหมือนเราสอนใครสักคนแล้วเลือกวิธีสอนแตกต่างกัน อันไหน work สุด
    - ชุดข้อมูลที่ใช้สำหรับการเรียนรู้ (โดย model) กล่าวคือ เพื่อให้เหมาะสมกับพารามิเตอร์กับ model ของ machine learning
test set นำไปใช้ทดสอบกับข้อมูลจริง ก่อนจะนำไปใช้งานจริงๆ
"""