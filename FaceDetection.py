import cv2 as cv


# 加载预训练的人脸识别模型
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 通过摄像头捕获视频
cap = cv.VideoCapture(0)

while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 将视频帧转换为灰度图像
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 在灰度图像上使用人脸级联分类器进行检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # 遍历检测到的人脸，绘制矩形框
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 显示结果
    cv.imshow('Face Detection', frame)

    # 检测键盘按键，如果按下 'q' 键则退出循环
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()

# 关闭所有窗口
cv.destroyAllWindows()
