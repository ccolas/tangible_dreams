import cv2
import numpy as np


def test_camera():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        # Read a frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame")
            break

        # Convert to grayscale to test what we'll use in CPPN
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Show both original and grayscale
        cv2.imshow('Original', frame)
        cv2.imshow('Grayscale', gray)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()