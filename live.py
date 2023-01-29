import cv2
import numpy as np
from keras.models import load_model

# MNIST
model = load_model('mnist_model.h5')

# webcam
cap = cv2.VideoCapture(0)

# predict window
cv2.namedWindow("Predict", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Predict", 28, 28)

# save video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (500, 500))

while True:
    # capture
    _, frame = cap.read()

    # break if fail to read webcam
    if not _:
        break

    # resize
    frame = cv2.resize(frame, (500, 500))

    # rectangle on frame
    cv2.rectangle(frame, (200, 200), (300, 300), (255, 0, 0), 2)

    # ROI
    roi = frame[200:300, 200:300]

    # resize ROI
    roi = cv2.resize(roi, (28, 28))

    # grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # reshape form (1, pixels)
    gray = gray.reshape(1, 28, 28, 1)

    # normalize
    gray = gray.astype('float32')
    gray /= 255

    # predict the digit
    prediction = model.predict(gray)
    digit = np.argmax(prediction)

    # display
    cv2.putText(roi, str(digit), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # display webcam
    cv2.imshow('Webcam', frame)
    # display predict
    cv2.imshow("Predict", roi)

    # video writer
    out.write(frame)

    # exit 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release
cap.release()
out.release()
cv2.destroyAllWindows()
