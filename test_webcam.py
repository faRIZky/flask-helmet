import cv2

def test_camera_index(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera with index {index}")
        return False
    print(f"Camera {index} opened successfully")
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f'Camera {index}', frame)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    return False

# Test indices
for i in range(5):  # Test indices from 0 to 4
    print(f"Testing camera index {i}")
    if test_camera_index(i):
        print(f"Camera index {i} works")
    else:
        print(f"Camera index {i} does not work")
