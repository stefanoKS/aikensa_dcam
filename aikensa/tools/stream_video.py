import cv2

# Open the camera with index 1
cap = cv2.VideoCapture(2)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully. Press 'q' to exit.")

# Stream the camera feed
while cap.isOpened():
    ret, frame = cap.read()
    
    # If a frame is successfully captured
    if ret:
        # Display the frame
        cv2.imshow("Camera Stream", frame)
        
        # Press 'q' to exit the streaming
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Could not read frame.")
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
