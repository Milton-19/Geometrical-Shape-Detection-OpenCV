import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        shape = ""
        formula = ""
        example = ""

        # Triangle
        if len(approx) == 3:
            shape = "Triangle"
            formula = "Area = 1/2 * base * height"
            example = "Example: Traffic sign"

        # Square or Rectangle
        elif len(approx) == 4:
            aspect_ratio = w / float(h)

            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
                formula = "Area = a^2"
                example = "Example: Floor tile"
            else:
                shape = "Rectangle"
                formula = "Area = l * b"
                example = "Example: Book / Door"

        # Circle
        elif len(approx) > 4:
            shape = "Circle"
            formula = "Area = pi * r^2"
            example = "Example: Wheel"

        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

        cv2.putText(frame, shape, (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

        cv2.putText(frame, formula, (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)

        cv2.putText(frame, example, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 2)

    cv2.imshow("Shape Detection with Details", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

