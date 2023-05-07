#Gradient Descent Algorithm (GD)
"""
ใช้หาจุดต่ำสุดหรือสูงสุดของฟังก์ชั่นที่กำหนดขึ้นมา
โดยใช้วิธีการวนหาค่าที่ทำให้ได้ค่าต่ำสุดจากการคำนวณความชัน ณ จุดที่เราอยู่แล้วพยายามเดินไปทางตรงข้ามกับกับความชันนั้น

พูดให้เห็นภาพคือ เหมือนการเดินวนหาเส้นทางจากยอดเขาที่สูงที่สุดแล้วลงไปใต้ดินที่ลึกที่สุด
Best case คือ จำนวนการวนรอบที่น้อยที่สุด ใช้เวลาในการหาเส้นทางแปบเดียวก็เจอทางที่ดี
Worst case คือ จำนวนวนรอบที่มากที่สุด ใช้เวลาในการหาเส้นทางนานกว่าจะเจอทางที่ดี

#รูปแบบการใช้งาน
    - คำนวนข้อมูลทั้งหมดทีเดียว (Full Batch Gradient Descent Algorithm)
    - คำนวนข้อมูลแค่บางส่วน (Stochastic Gradient Descent Algorithm)

    
#จาก LAB (ML3.1_Ex_Weather) 
ตัวอย่างเนื้อหา Linear Regression การหาอุณหภุมิสูงสุดต่ำสุด เรื่องการวัดประสิทธิภาพโมเดลยังมี Error อยู่พอสมควร
ซึ่งวัดได้จากค่า MSE (Mean Square Error) ตัว Gradient Descent 
จะช่วยทำให้โมเดลของเรามีค่า Loss Function ลงไปยังจุดที่ใกล้ 0 มากที่สุด (จุดต่ำสุด) เพื่อให้ได้โมเดลที่มีประสิทธิภาพมากยิ่งขึ้น

#ทำไมต้องใช้ Stochastic Gradient Descent Algorithm
https://miro.medium.com/v2/resize:fit:720/format:webp/0*YVSXUv4zPZ2-Tjp0.png
    อันนี้เข้าใจเองนะว่าเรามีคำตอบอยู่แล้ว มีเฉลยอยู่แล้วที่เราเตรียมไว้ 
    แค่เอามาทดสอบ model เฉยๆ ว่ามันตรงกับเฉลยของเราไหม ถ้าตรงกันก็น่าจะนำไปใช้ต่อได้
"""