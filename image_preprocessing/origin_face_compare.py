import os
import cv2

def resize_image_to_screen(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(image, None, fx=scale, fy=scale), scale

def mouse_callback(event, x, y, flags, param):
    global selection_rect, drawing, updated

    if event == cv2.EVENT_LBUTTONDOWN:
        selection_rect[0] = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            selection_rect[1] = (x, y)
            updated = True
    elif event == cv2.EVENT_LBUTTONUP:
        selection_rect[1] = (x, y)
        drawing = False
        updated = True

def save_cropped_face(original_image, resized_image, rect, output_dir, image_name, scale):
    x1, y1, x2, y2 = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
    original_rect = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
    cropped_image = original_image[original_rect[1]:original_rect[3], original_rect[0]:original_rect[2]]
    
    if not cropped_image.size == 0:  # Add this line to check if cropped_image is not empty
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_face.jpg"), cropped_image)

image_folder = "data/girl/"
detected_faces_folder = "data/detected_faces/"
output_dir = "data/cropped_faces/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
detected_faces_files = [os.path.join(detected_faces_folder, f) for f in os.listdir(detected_faces_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

#selection_rect = [None, None]
drawing = False
updated = True
recropped = False
remaining_images = len(detected_faces_files)
for detected_face_path in detected_faces_files:
    print(f"Remaining images: {remaining_images}")
    remaining_images -= 1
    selection_rect = [None, None]
    image_name = os.path.basename(detected_face_path).replace("_face.jpg", "")
    image_path = os.path.join(image_folder, image_name)
    
    original_image = cv2.imread(image_path)
    resized_image, scale = resize_image_to_screen(original_image)
    detected_face = cv2.imread(detected_face_path)
    detected_face_resized, _ = resize_image_to_screen(detected_face, max_width=300, max_height=300)

    cv2.namedWindow('image')
    cv2.namedWindow('detected_face')
    cv2.moveWindow('detected_face', 850, 100)       
    cv2.setMouseCallback('image', mouse_callback)

    while True:
        display_image = resized_image.copy()
        if selection_rect[0] and selection_rect[1]:
            cv2.rectangle(display_image, selection_rect[0], selection_rect[1], (0, 255, 0), 2)
        cv2.imshow('image', display_image)
        cv2.imshow('detected_face', detected_face_resized)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if selection_rect[0] and selection_rect[1]:
                save_cropped_face(original_image, resized_image, selection_rect, output_dir, image_name, scale)
                os.remove(detected_face_path)
                break
            else:
                os.rename(detected_face_path, os.path.join(output_dir, f"{image_name}_face.jpg"))
                break

        elif key == ord('d'):
            os.remove(detected_face_path)
            break
        
    cv2.destroyAllWindows()