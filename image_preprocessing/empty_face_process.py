import os
import cv2

# Functions
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
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_face.jpg"), cropped_image)

# Variables
image_folder = "data/no_faces/"
output_dir = "data/cropped_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

selection_rect = [None, None]
drawing = False
updated = True
remaining_images = len(image_files)
# Main loop
for image_path in image_files:
    print(f"Remaining images: {remaining_images}")
    remaining_images -= 1
    selection_rect = [None, None]
    original_image = cv2.imread(image_path)
    resized_image, scale = resize_image_to_screen(original_image)

    image_name = os.path.basename(image_path)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    while True:
        if updated:
            display_image = resized_image.copy()
            if selection_rect[0] and selection_rect[1]:
                cv2.rectangle(display_image, selection_rect[0], selection_rect[1], (0, 255, 0), 2)
            cv2.imshow('image', display_image)
            updated = False

        key = cv2.waitKey(50) & 0xFF

        if key == ord('s'):
            if selection_rect[0] and selection_rect[1]:
                save_cropped_face(original_image, resized_image, selection_rect, output_dir, image_name, scale)
                os.remove(image_path)
                break

        elif key == ord('d'):
            os.remove(image_path)
            break

    cv2.destroyAllWindows()
    updated = True

cv2.destroyAllWindows()
