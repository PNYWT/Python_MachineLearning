# pip install -U scikit-learn
# https://kongruksiam.medium.com/สรุป-machine-learning-ep-1-ว่าด้วยเรื่องชุดข้อมูล-dataset-f3167b829406

from sklearn import datasets

iris_Datasets = datasets.load_iris()

"""
Key of iris_Datasets
    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

    target -> สายพันธุ์ เป็น รหัสตัวเลข
    target_names -> สายพันธุ์ เป็น ชื่อ
    DESCR -> รายละเอียด คำอธิบาย
    feature_names -> ข้อมูล ความกว้าง ยาว สูง  หน่วย cm ของแต่ละสายพันธุ์
    data -> ข้อมูลที่บันทึก ความกว้าง ยาว สูง เป็นตัวเลข ของแต่ละสายพันธุ์ 
"""
# print(iris_Datasets["data"][0:10])
# print(iris_Datasets.target_names)
print(iris_Datasets.data[0:10])
