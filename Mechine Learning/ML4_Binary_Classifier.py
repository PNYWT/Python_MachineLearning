#https://kongruksiam.medium.com/สรุป-machine-learning-ep-4-ตัวจำแนกแบบไบรารี่-binary-classifier-6ebc8e1a5e61
"""
ตัวจำแนกแบบไบนารี (Binary Classifier) -> เป็นวิธีการแบ่งข้อมูลออกเป็น 2 กลุ่ม

ภายใน LAB จะยกตัวอย่างเพื่อให้เห็นการแบ่งกลุ่มของข้อมูลโดยใช้ MNIST Dataset ว่าข้อมูลกลุ่มใดแสดงกลุ่มตัวเลข 0-9 
โดยข้อมูลทั้งหมดจะมี 70,000 ชุดจะต้องเขียนโปรแกรมแบ่งข้อมูลออกเป็น 2 ส่วนได้แก่
    - Training Set 60,000 ชุด
        ข้อมูลชุดเรียนรู้ (Training Set) จัดเรียงและแบ่งกลุ่มเป็น Class 0 — Class 9
    - Test Set 10,000 ชุด
        ข้อมูลชุดทดสอบ (Test Set) จัดเรียงและแบ่งกลุ่มเป็น Class 0 — Class 9
"""

