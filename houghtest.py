# import cv2
# import numpy as np
# from collections import Counter
# import tensorflow as tf
# import json
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
# from tensorflow.keras.applications import MobileNetV2

# # Function to detect the largest circle by consistent radius
# def detect_consistent_largest_circle(image, iterations=10):
#     detected_circles = []
    
#     for i in range(iterations):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image_height, image_width = image.shape[:2]

#         gray_blurred = cv2.GaussianBlur(gray, (15, 15), 2)
        
#         circles = cv2.HoughCircles(
#             gray_blurred,
#             cv2.HOUGH_GRADIENT,
#             dp=1.6,
#             minDist=100,
#             param1=100,
#             param2=100,
#             minRadius=int(min(image_width, image_height) * 0.1),
#             maxRadius=int(min(image_width, image_height) * 0.5)
#         )
        
#         if circles is not None:
#             circles = np.round(circles[0, :]).astype("int")
#             detected_circles.extend(circles)

#     if detected_circles:
#         radius_counter = Counter([circle[2] for circle in detected_circles])
#         most_common_radius = radius_counter.most_common(1)[0][0]
#         largest_circle = max([circle for circle in detected_circles if circle[2] == most_common_radius], key=lambda x: x[2])
#         print(f"Detected consistent largest circle: center=({largest_circle[0]}, {largest_circle[1]}), radius={largest_circle[2]}")
#         return largest_circle
#     else:
#         print("No circles detected.")
#         return None

# # Automatically load MobileNetV2 model if not present
# print("Downloading and loading the MobileNetV2 model...")
# model = MobileNetV2(weights='imagenet')
# print("Model loaded successfully!")

# # Load and process image
# image_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\fan8.png'
# output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\fan8.png'
# json_output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\fan8.json'
# image = cv2.imread(image_path)

# if image is None:
#     print("Error: Image not found or cannot be opened.")
# else:
#     largest_circle = detect_consistent_largest_circle(image)

#     if largest_circle is not None:
#         x, y, r = map(int, largest_circle)  # Ensure all values are int
#         output = image.copy()

#         # Extract the region inside the circle
#         mask = np.zeros_like(image)
#         cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
#         circle_region = cv2.bitwise_and(image, mask)
#         circle_crop = circle_region[y-r:y+r, x-r:x+r]

#         # Resize to model input size and preprocess
#         model_input_size = (224, 224)
#         circle_resized = cv2.resize(circle_crop, model_input_size)
#         circle_preprocessed = preprocess_input(np.expand_dims(circle_resized, axis=0))

#         # Classify the region using MobileNetV2
#         predictions = model.predict(circle_preprocessed)
#         decoded_predictions = decode_predictions(predictions, top=1)[0][0]
#         label = f"{decoded_predictions[1]}: {decoded_predictions[2]*100:.2f}%"

#         # Save label in JSON format
#         classification_result = {
#             "label": decoded_predictions[1],
#             "confidence": float(decoded_predictions[2] * 100),  # Convert to float
#             "center": {"x": int(x), "y": int(y)},
#             "radius": int(r)  # Ensure radius is an int
#         }
#         with open(json_output_path, 'w') as json_file:
#             json.dump(classification_result, json_file, indent=4)
        
#         # Draw the classification label
#         cv2.putText(output, label, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Draw the detected circle
#         cv2.circle(output, (x, y), r, (0, 255, 0), 4)

#        # Add lines r, a, v
#         # Line for 'r' (radius) - Bold line and label
#         end_point_r = (x - r, y)
#         cv2.line(output, (x, y), end_point_r, (0, 255, 255), 4)  # Increased thickness to 4
#         mid_point_r = ((x + end_point_r[0]) // 2, (y + end_point_r[1]) // 2)
#         cv2.putText(output, 'r', (mid_point_r[0] - 10, mid_point_r[1] + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # Larger font and thicker

#         # Line for 'a' (arrow inward at 45 degrees) - Bold line and label
#         angle_a = 45
#         start_point_a = (
#             int(x + r * np.cos(np.deg2rad(angle_a))),
#             int(y - r * np.sin(np.deg2rad(angle_a)))
#         )
#         end_point_a = (
#             int(start_point_a[0] - 0.5 * r * np.cos(np.deg2rad(angle_a))),
#             int(start_point_a[1] + 0.5 * r * np.sin(np.deg2rad(angle_a)))
#         )
#         cv2.arrowedLine(output, start_point_a, end_point_a, (0, 255, 255), 4, tipLength=0.3)  # Increased thickness and tip
#         mid_point_a = ((start_point_a[0] + end_point_a[0]) // 2, (start_point_a[1] + end_point_a[1]) // 2)
#         cv2.putText(output, 'a', (mid_point_a[0] - 10, mid_point_a[1] + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # Larger font and thicker

#         # Line for 'v' (arrow outward) - Bold line and label
#         angle_v = angle_a + 270
#         end_point_v = (
#             int(start_point_a[0] + 0.5 * r * np.cos(np.deg2rad(angle_v))),
#             int(start_point_a[1] - 0.5 * r * np.sin(np.deg2rad(angle_v)))
#         )
#         cv2.arrowedLine(output, start_point_a, end_point_v, (0, 255, 255), 4, tipLength=0.3)  # Increased thickness and tip
#         mid_point_v = ((start_point_a[0] + end_point_v[0]) // 2, (start_point_a[1] + end_point_v[1]) // 2)
#         cv2.putText(output, 'v', (mid_point_v[0] - 10, mid_point_v[1] + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # Larger font and thicker


#         # Save the output image
#         cv2.imwrite(output_path, output)
#         print(f"Output image saved as {output_path}")
#         print(f"Classification result saved as {json_output_path}")
#         print(f"Classified object: {label}")
#     else:
#         print("No circle was reliably detected as the largest circle.")


#### EfficientNetB7

# import cv2
# import numpy as np
# from collections import Counter
# import tensorflow as tf
# import json
# from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
# from tensorflow.keras.applications import EfficientNetB7

# # Function to detect the largest circle by consistent radius
# def detect_consistent_largest_circle(image, iterations=10):
#     detected_circles = []
    
#     for i in range(iterations):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image_height, image_width = image.shape[:2]

#         gray_blurred = cv2.GaussianBlur(gray, (15, 15), 2)
        
#         circles = cv2.HoughCircles(
#             gray_blurred,
#             cv2.HOUGH_GRADIENT,
#             dp=1.6,
#             minDist=100,
#             param1=100,
#             param2=100,
#             minRadius=int(min(image_width, image_height) * 0.1),
#             maxRadius=int(min(image_width, image_height) * 0.5)
#         )
        
#         if circles is not None:
#             circles = np.round(circles[0, :]).astype("int")
#             detected_circles.extend(circles)

#     if detected_circles:
#         radius_counter = Counter([circle[2] for circle in detected_circles])
#         most_common_radius = radius_counter.most_common(1)[0][0]
#         largest_circle = max([circle for circle in detected_circles if circle[2] == most_common_radius], key=lambda x: x[2])
#         print(f"Detected consistent largest circle: center=({largest_circle[0]}, {largest_circle[1]}), radius={largest_circle[2]}")
#         return largest_circle
#     else:
#         print("No circles detected.")
#         return None

# # Automatically load EfficientNetB7 model if not present
# print("Downloading and loading the EfficientNetB7 model...")
# model = EfficientNetB7(weights='imagenet')
# print("Model loaded successfully!")

# # Load and process image
# image_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\fan6.jpeg'
# output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\fan6.jpeg'
# json_output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\fan6.json'
# image = cv2.imread(image_path)

# if image is None:
#     print("Error: Image not found or cannot be opened.")
# else:
#     largest_circle = detect_consistent_largest_circle(image)

#     if largest_circle is not None:
#         x, y, r = map(int, largest_circle)  # Ensure all values are int
#         output = image.copy()

#         # Define the region of interest (ROI) around the circle
#         roi_x1 = max(0, x - r)
#         roi_y1 = max(0, y - r)
#         roi_x2 = min(image.shape[1], x + r)
#         roi_y2 = min(image.shape[0], y + r)
#         roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

#         # Resize to model input size and preprocess
#         model_input_size = (600, 600)  # EfficientNetB7 input size
#         roi_resized = cv2.resize(roi, model_input_size)
#         roi_preprocessed = preprocess_input(np.expand_dims(roi_resized, axis=0))

#         # Classify the region using EfficientNetB7
#         predictions = model.predict(roi_preprocessed)
#         decoded_predictions = decode_predictions(predictions, top=1)[0][0]
#         label = decoded_predictions[1]  # Only the object name

#         # Save label in JSON format
#         classification_result = {
#             "label": decoded_predictions[1],
#             "confidence": float(decoded_predictions[2] * 100),
#             "center": {"x": int(x), "y": int(y)},
#             "radius": int(r)
#         }
#         with open(json_output_path, 'w') as json_file:
#             json.dump(classification_result, json_file, indent=4)

#         # Draw the classification label (object name only)
#         cv2.putText(output, label, (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         # Draw the detected circle
#         cv2.circle(output, (x, y), r, (0, 255, 0), 4)

#         # Add lines r, a, v
#         # Line for 'r' (radius) - Bold line and label
#         end_point_r = (x - r, y)
#         cv2.line(output, (x, y), end_point_r, (0, 255, 255), 4)  # Increased thickness to 4
#         mid_point_r = ((x + end_point_r[0]) // 2, (y + end_point_r[1]) // 2)
#         cv2.putText(output, 'r', (mid_point_r[0] - 10, mid_point_r[1] + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # Larger font and thicker

#         # Line for 'a' (arrow inward at 45 degrees) - Bold line and label
#         angle_a = 45
#         start_point_a = (
#             int(x + r * np.cos(np.deg2rad(angle_a))),
#             int(y - r * np.sin(np.deg2rad(angle_a)))
#         )
#         end_point_a = (
#             int(start_point_a[0] - 0.5 * r * np.cos(np.deg2rad(angle_a))),
#             int(start_point_a[1] + 0.5 * r * np.sin(np.deg2rad(angle_a)))
#         )
#         cv2.arrowedLine(output, start_point_a, end_point_a, (0, 255, 255), 4, tipLength=0.3)  # Increased thickness and tip
#         mid_point_a = ((start_point_a[0] + end_point_a[0]) // 2, (start_point_a[1] + end_point_a[1]) // 2)
#         cv2.putText(output, 'a', (mid_point_a[0] - 10, mid_point_a[1] + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # Larger font and thicker

#         # Line for 'v' (arrow outward) - Bold line and label
#         angle_v = angle_a + 270
#         end_point_v = (
#             int(start_point_a[0] + 0.5 * r * np.cos(np.deg2rad(angle_v))),
#             int(start_point_a[1] - 0.5 * r * np.sin(np.deg2rad(angle_v)))
#         )
#         cv2.arrowedLine(output, start_point_a, end_point_v, (0, 255, 255), 4, tipLength=0.3)  # Increased thickness and tip
#         mid_point_v = ((start_point_a[0] + end_point_v[0]) // 2, (start_point_a[1] + end_point_v[1]) // 2)
#         cv2.putText(output, 'v', (mid_point_v[0] - 10, mid_point_v[1] + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # Larger font and thicker

#         # Save the output image
#         cv2.imwrite(output_path, output)
#         print(f"Output image saved as {output_path}")
#         print(f"Classification result saved as {json_output_path}")
#         print(f"Classified object: {label}")
#     else:
#         print("No circle was reliably detected as the largest circle.")

####DeepSeek

# import cv2
# import numpy as np
# from collections import defaultdict
# import tensorflow as tf
# import json
# from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
# from tensorflow.keras.applications import EfficientNetB7

# def detect_consistent_circle(image, iterations=10):
#     detected_circles = []
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape
#     min_dim = min(height, width)
    
#     # Contrast Limited Adaptive Histogram Equalization
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
#     for _ in range(iterations):
#         enhanced_gray = clahe.apply(gray)
        
#         # Dynamic blur based on image size
#         kernel_size = int(min_dim * 0.02) // 2 * 2 + 1  # Ensure odd
#         blurred = cv2.GaussianBlur(enhanced_gray, (kernel_size, kernel_size), 0)
        
#         # Apply Adaptive Thresholding
#         thresh = cv2.adaptiveThreshold(
#             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY, 11, 2
#         )
        
#         # Use a fixed value for param1 in HoughCircles
#         high_thresh = 100  # Fixed value for edge detection
        
#         # Dynamic Hough parameters
#         params = {
#             'dp': 1.6,
#             'minDist': int(min_dim * 0.2),
#             'param1': high_thresh,  # Upper threshold for edge detection
#             'param2': 30,  # Accumulator threshold (fixed for simplicity)
#             'minRadius': int(min_dim * 0.05),
#             'maxRadius': int(min_dim * 0.45)
#         }
        
#         circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **params)
        
#         if circles is not None:
#             circles = np.round(circles[0, :]).astype("int")
#             detected_circles.extend(circles)
    
#     # Cluster circles using geometric hashing
#     circle_groups = defaultdict(list)
#     for (x, y, r) in detected_circles:
#         key = (x//50, y//50, r//20)  # Spatial and radial binning
#         circle_groups[key].append((x, y, r))
    
#     # Find the most consistent cluster
#     if circle_groups:
#         best_group = max(circle_groups.values(), key=len)
#         avg_circle = np.mean(best_group, axis=0).astype(int)
#         return tuple(avg_circle)
#     return None

# # Load model
# model = EfficientNetB7(weights='imagenet')

# # Process image
# image_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\S__11509778_0.jpg'
# output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\S__11509778_0.jpg'
# json_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\S__11509778_0.json'


# image = cv2.imread(image_path)
# if image is None:
#     raise ValueError("Could not read image")

# circle = detect_consistent_circle(image)

# if circle:
#     x, y, r = circle
#     output = image.copy()
    
#     # Classification ROI processing
#     roi = image[y-r:y+r, x-r:x+r]
#     roi = cv2.resize(roi, (600, 600))
#     roi_preprocessed = preprocess_input(np.expand_dims(roi, axis=0))
    
#     # Prediction
#     preds = model.predict(roi_preprocessed)
#     label = decode_predictions(preds, top=1)[0][0][1]
    
#     # Save results
#     result = {
#         "label": label,
#         "confidence": float(preds[0][np.argmax(preds)]),
#         "geometry": {"center": (int(x), int(y)), "radius": int(r)}
#     }
    
#     with open(json_path, 'w') as f:
#         json.dump(result, f)
    
#     # Visualization
#     cv2.circle(output, (x, y), r, (0, 255, 0), 4)

#     # x_offset = -35
#     # y_offset = -35
#     # r_offset = -35
#     # cv2.circle(output, (x + x_offset, y + y_offset), r + r_offset, (0, 255, 0), 4)

#     cv2.putText(output, label, (x-r, y-r-10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     cv2.imwrite(output_path, output)
# else:
#     print("No circles detected")

###DeepSeek

# import cv2
# import numpy as np
# from collections import defaultdict
# import json

# from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
# from tensorflow.keras.applications import EfficientNetB7

# def draw_text_with_background_centered(
#     image,
#     text,
#     center_pt,           # The desired center of the text+background
#     font=cv2.FONT_HERSHEY_SIMPLEX,
#     font_scale=1.0,
#     color=(0, 0, 255),
#     thickness=2,
#     bg_color=(255, 255, 255),
#     padding=10
# ):
#     """
#     Draw text such that 'center_pt' is the center of the 
#     padded white background. The text itself is also centered 
#     inside that rectangle.
#     """
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

#     # 1) Compute the rectangle corners to center around 'center_pt'
#     rect_x1 = int(center_pt[0] - (text_width / 2) - padding)
#     rect_y1 = int(center_pt[1] - (text_height / 2) - padding)
#     rect_x2 = int(center_pt[0] + (text_width / 2) + padding)
#     rect_y2 = int(center_pt[1] + (text_height / 2) + baseline + padding)

#     # 2) Draw the white rectangle
#     cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

#     # 3) Place the text so that its center is near 'center_pt'
#     text_x = int(center_pt[0] - (text_width / 2))
#     # 'cv2.putText' draws text baseline at 'y' => shift up by half text height
#     text_y = int(center_pt[1] + (text_height / 2))

#     cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)


# def detect_consistent_circle(image, iterations=10):
#     """
#     Mencoba mendeteksi lingkaran secara berulang (HoughCircles) lalu
#     memilih yang paling sering muncul. Bisa Anda sesuaikan lagi parameternya.
#     """
#     from collections import defaultdict
    
#     detected_circles = []
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape
#     min_dim = min(height, width)

#     # Optional: CLAHE untuk meningkatkan kontras
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
#     for _ in range(iterations):
#         enhanced_gray = clahe.apply(gray)
#         kernel_size = int(min_dim * 0.02) // 2 * 2 + 1  # kernel blur, pastikan ganjil
#         blurred = cv2.GaussianBlur(enhanced_gray, (kernel_size, kernel_size), 0)
        
#         # Threshold adaptif
#         thresh = cv2.adaptiveThreshold(
#             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY, 11, 2
#         )
        
#         params = {
#             'dp': 1.6,
#             'minDist': int(min_dim * 0.2),
#             'param1': 100,
#             'param2': 30,
#             'minRadius': int(min_dim * 0.05),
#             'maxRadius': int(min_dim * 0.45)
#         }
        
#         circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **params)
        
#         if circles is not None:
#             circles = np.round(circles[0, :]).astype("int")
#             detected_circles.extend(circles)
    
#     # Kelompokkan lingkaran mirip
#     circle_groups = defaultdict(list)
#     for (x, y, r) in detected_circles:
#         key = (x//50, y//50, r//20)
#         circle_groups[key].append((x, y, r))
    
#     # Pilih group terbanyak, lalu ambil rata-rata
#     if circle_groups:
#         best_group = max(circle_groups.values(), key=len)
#         avg_circle = np.mean(best_group, axis=0).astype(int)
#         return tuple(avg_circle)
#     return None

# def draw_curved_arrow(
#     img,
#     center,        # (cx, cy)
#     radius,        # jari-jari arc
#     start_angle,   # sudut awal arc
#     end_angle,     # sudut akhir arc
#     color=(0,165,255),
#     thickness=4,
#     label="ω",
#     tip_len=20,
#     font_scale=1.0
# ):
#     """
#     Menggambar busur (arc) lalu ujung panah manual (2 garis).
#     0° = jam 3, sudut bertambah searah jarum jam di OpenCV.
#     """
#     # 1) Gambar arc
#     cv2.ellipse(
#         img,
#         center,
#         (radius, radius),
#         0,
#         start_angle,
#         end_angle,
#         color,
#         thickness
#     )

#     # 2) Titik akhir arc
#     end_rad = np.deg2rad(end_angle)
#     end_x = center[0] + radius * np.cos(end_rad)
#     end_y = center[1] + radius * np.sin(end_rad)

#     # 3) Ujung panah (2 garis kecil)
#     tangent_angle = end_rad + np.deg2rad(90)  # tangensial
#     spread = np.deg2rad(30)
    
#     tip1 = (
#         end_x + tip_len * np.cos(tangent_angle + spread),
#         end_y + tip_len * np.sin(tangent_angle + spread)
#     )
#     tip2 = (
#         end_x + tip_len * np.cos(tangent_angle - spread),
#         end_y + tip_len * np.sin(tangent_angle - spread)
#     )

#     cv2.line(img, (int(end_x), int(end_y)), (int(tip1[0]), int(tip1[1])), color, thickness)
#     cv2.line(img, (int(end_x), int(end_y)), (int(tip2[0]), int(tip2[1])), color, thickness)

#     # 4) Label di sekitar tengah arc
    
#     mid_angle = (start_angle + end_angle) / 2.0
#     mid_rad = np.deg2rad(mid_angle)
#     mid_x = center[0] + radius * np.cos(mid_rad)
#     mid_y = center[1] + radius * np.sin(mid_rad)
#     offset = 0.2 * radius  # Jarak label dari arc

#     label_center_w = (
#         int(mid_x),
#         int(mid_y- offset - 20)
#     )
#     draw_text_with_background_centered(
#         img,
#         label,  # e.g. "ω"
#         label_center_w,
#         font_scale=font_scale,
#         color=(0,0,225),  # or (0,0,255)
#         thickness=int(thickness),
#         padding=5
#     )

# # -- MAIN / DEMO ----------------------------------------------------------------

# model = EfficientNetB7(weights='imagenet')

# # Ganti path Anda sendiri
# image_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\S__11509778_0.jpg'
# output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\S__11509778_0.jpg'
# json_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\S__11509778_0.json'

# image = cv2.imread(image_path)
# if image is None:
#     raise ValueError("Could not read image")

# circle = detect_consistent_circle(image)
# if circle:
#     x, y, r = circle
#     output = image.copy()
    
#     # Contoh klasifikasi ROI (opsional)
#     roi = image[y-r:y+r, x-r:x+r]
#     roi = cv2.resize(roi, (600, 600))
#     roi_preprocessed = preprocess_input(np.expand_dims(roi, axis=0))
#     preds = model.predict(roi_preprocessed)
#     top_label = decode_predictions(preds, top=1)[0][0][1]
    
#     # Simpan JSON (opsional)
#     result = {
#         "label": top_label,
#         "confidence": float(preds[0][np.argmax(preds)]),
#         "geometry": {"center": (int(x), int(y)), "radius": int(r)}
#     }
#     with open(json_path, 'w') as f:
#         json.dump(result, f)
    
#     # ----------------------------
#     # 1) Buat scale factor
#     #    Semakin besar r, semakin besar garis, teks, dsb.
#     # ----------------------------
#     scale_factor = r / 100.0  # Silakan ubah rumusnya sesuai selera
    
#     # Minimal thickness = 2, agar tidak terlalu tipis
#     thickness = max(1.5, int(scale_factor * 3))  
#     # Minimal font_scale = 0.5, agar tidak terlalu kecil
#     font_scale = max(1, scale_factor * 0.5)
#     # Panjang ujung panah
#     arrow_tip_len = int(r * 0.2)
    
#     # ----------------------------
#     # 2) Gambar lingkaran utama
#     # ----------------------------
#     cv2.circle(output, (x, y), r, (0, 255, 0), thickness)

#     # ----------------------------
#     # 3) Garis radius (r)
#     # ----------------------------
#     end_point_r = (x - r, y)
#     cv2.line(output, (x, y), end_point_r, (0, 255, 255), thickness)
    
#     # Adjusted r label position (closer to line)
#     mid_point_r = (
#         (x + end_point_r[0]) // 2,
#         (y + end_point_r[1]) // 2
#     )
#     label_center_r = (
#         mid_point_r[0] + int(15 * scale_factor),
#         mid_point_r[1] - int(20 * scale_factor)
#     )
#     draw_text_with_background_centered(
#         output,
#         'r',
#         label_center_r,  # This is now the center of the background
#         font_scale=font_scale,
#         color=(0, 0, 255),
#         thickness=int(thickness),
#         bg_color=(255, 255, 255),
#         padding=5
#     )

#     # ----------------------------
#     # 4) Garis a (arrowedLine) with consistent styling
#     # ----------------------------
#     angle_a = 45
#     start_point_a = (
#         int(x + r * np.cos(np.deg2rad(angle_a))),
#         int(y - r * np.sin(np.deg2rad(angle_a)))
#     )
#     end_point_a = (
#         int(start_point_a[0] - 0.5 * r * np.cos(np.deg2rad(angle_a))),
#         int(start_point_a[1] + 0.5 * r * np.sin(np.deg2rad(angle_a)))
#     )
#     cv2.arrowedLine(
#         output,
#         start_point_a,
#         end_point_a,
#         (0, 255, 255),  # Yellow color
#         thickness=thickness,
#         tipLength=0.3
#     )

#     # Adjusted a label position
#     mid_point_a = (
#         (start_point_a[0] + end_point_a[0]) // 2,
#         (start_point_a[1] + end_point_a[1]) // 2
#     )

#     label_center_a = (
#         mid_point_a[0] - int(15 * scale_factor),
#         mid_point_a[1] - int(20 * scale_factor)
#     )

#     draw_text_with_background_centered(
#         output,
#         'a',
#         label_center_a,
#         font_scale=font_scale,
#         color=(0, 0, 255),
#         thickness=int(thickness),
#         padding=5
#     )


#     # ----------------------------
#     # 5) Panah W (busur melengkung) with matching style
#     # ----------------------------
#     startAngle = -210
#     endAngle = -270
#     curved_radius = int(r * 0.7)

#     draw_curved_arrow(
#         output,
#         center=(x, y),
#         radius=curved_radius,
#         start_angle=startAngle,
#         end_angle=endAngle,
#         color=(0, 255, 255),  # Changed to yellow to match
#         thickness=thickness,
#         label="w",
#         tip_len=int(r * 0.3),  # Proportional tip length (matches arrowedLine's 0.3 ratio)
#         font_scale=font_scale
#     )

#     # ----------------------------
#     # 6) Simpan hasil
#     # ----------------------------
#     cv2.imwrite(output_path, output)
#     print("Done. Result saved to:", output_path)
# else:
#     print("No circles detected")

# #####################
# ######################
# #######################
# #######################
# ########################3
# import os
# import cv2
# import numpy as np
# import json
# from collections import defaultdict
# from tensorflow.keras.applications import EfficientNetB7
# from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# #####################################
# # 1) UTILITY: Overlays an RGBA image
# #####################################
# def overlay_image_alpha(bg, fg, x, y):
#     """
#     Overlay 'fg' onto 'bg' at (x, y).
#     If 'fg' has 4 channels => alpha blend each pixel.
#     """
#     fh, fw = fg.shape[:2]
#     for row in range(fh):
#         for col in range(fw):
#             # Check bounds
#             if (y + row >= bg.shape[0]) or (x + col >= bg.shape[1]):
#                 continue
#             alpha = 1.0
#             # If 'fg' is RGBA => the 4th channel is alpha
#             if fg.shape[2] == 4:
#                 alpha = fg[row, col, 3] / 255.0
#             for c in range(3):  # B, G, R channels
#                 bg[y + row, x + col, c] = (
#                     alpha * fg[row, col, c] +
#                     (1 - alpha) * bg[y + row, x + col, c]
#                 )

# ##############################################
# # 2) CENTERED TEXT: for "r" and "a" labels
# ##############################################
# def draw_text_with_background_centered(
#     image,
#     text,
#     center_pt,
#     font=cv2.FONT_HERSHEY_SIMPLEX,
#     font_scale=1.0,
#     color=(0, 0, 255),
#     thickness=2,
#     bg_color=(255, 255, 255),
#     padding=10
# ):
#     """
#     Draw text so 'center_pt' is the center of the padded white rectangle,
#     with text also centered inside that rectangle.
#     """
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

#     rect_x1 = int(center_pt[0] - (text_width / 2) - padding)
#     rect_y1 = int(center_pt[1] - (text_height / 2) - padding)
#     rect_x2 = int(center_pt[0] + (text_width / 2) + padding)
#     rect_y2 = int(center_pt[1] + (text_height / 2) + baseline + padding)

#     # Draw white rectangle
#     cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

#     text_x = int(center_pt[0] - (text_width / 2))
#     text_y = int(center_pt[1] + (text_height / 2))

#     cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

# # ##############################################
# # # 3) DETECT: Attempt repeated circle detection
# # ##############################################
# def detect_consistent_circle(image, iterations=10):
#     """
#     Repeatedly run HoughCircles, pick the circle that appears most frequently.
#     """
#     detected_circles = []
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape
#     min_dim = min(height, width)

#     # Contrast Limited Adaptive Histogram Equalization
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#     for _ in range(iterations):
#         enhanced_gray = clahe.apply(gray)
#         kernel_size = int(min_dim * 0.02) // 2 * 2 + 1
#         blurred = cv2.GaussianBlur(enhanced_gray, (kernel_size, kernel_size), 0)

#         _ = cv2.adaptiveThreshold(
#             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY, 11, 2
#         )

#         params = {
#             'dp': 1.6,
#             'minDist': int(min_dim * 0.2),
#             'param1': 100,
#             'param2': 30,
#             'minRadius': int(min_dim * 0.05),
#             'maxRadius': int(min_dim * 0.45)
#         }
#         circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **params)
#         if circles is not None:
#             circles = np.round(circles[0, :]).astype("int")
#             detected_circles.extend(circles)

#     # Group similar circles
#     circle_groups = defaultdict(list)
#     for (cx, cy, rr) in detected_circles:
#         key = (cx // 50, cy // 50, rr // 20)
#         circle_groups[key].append((cx, cy, rr))

#     if circle_groups:
#         best_group = max(circle_groups.values(), key=len)
#         avg_circle = np.mean(best_group, axis=0).astype(int)
#         return tuple(avg_circle)
#     return None

# # ##############################################
# # 4) ARC + ARROW TIP, Resized Omega Symbol
# ##############################################
# def draw_curved_arrow(
#     img,
#     center,
#     radius,
#     start_angle,
#     end_angle,
#     color=(0, 165, 255),
#     thickness=4,
#     tip_len=20,
#     font_scale=1.0,
#     omega_symbol_path=r".\image_raw\Greek_lc_omega.png"
# ):
#     """
#     Draw an arc + arrow tips. Then compute a bounding-box as if
#     we were drawing text 'dummy' => resize 'omega_symbol.png'
#     to exactly fill that rectangle, and overlay it.
#     """

#     # 1) Arc
#     cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

#     # 2) Arrow tip lines
#     end_rad = np.deg2rad(end_angle)
#     end_x = center[0] + radius * np.cos(end_rad)
#     end_y = center[1] + radius * np.sin(end_rad)

#     tangent_angle = end_rad + np.deg2rad(90)
#     spread = np.deg2rad(30)
#     tip1 = (
#         end_x + tip_len * np.cos(tangent_angle + spread),
#         end_y + tip_len * np.sin(tangent_angle + spread)
#     )
#     tip2 = (
#         end_x + tip_len * np.cos(tangent_angle - spread),
#         end_y + tip_len * np.sin(tangent_angle - spread)
#     )

#     cv2.line(img, (int(end_x), int(end_y)), (int(tip1[0]), int(tip1[1])), color, thickness)
#     cv2.line(img, (int(end_x), int(end_y)), (int(tip2[0]), int(tip2[1])), color, thickness)

#     # 3) Decide where to place the bounding box => arc midpoint
#     mid_angle = (start_angle + end_angle) / 2.0
#     mid_rad = np.deg2rad(mid_angle)
#     mid_x = int(center[0] + radius * np.cos(mid_rad))
#     mid_y = int(center[1] + radius * np.sin(mid_rad))

#     # We'll offset it upward a bit
#     offset = 0.2 * radius
#     rect_center_x = mid_x
#     rect_center_y = int(mid_y - offset - 30)

#     # 4) Let's measure a bounding box as if text = "dummy"
#     text = "x"
#     (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

#     # compute rectangle corners around rect_center
#     padding = 10
#     rect_x1 = int(rect_center_x - (text_width / 2) - padding)
#     rect_y1 = int(rect_center_y - (text_height / 2) - padding)
#     rect_x2 = int(rect_center_x + (text_width / 2) + padding)
#     rect_y2 = int(rect_center_y + (text_height / 2) + baseline + padding)

#     # compute final bounding box size
#     box_w = rect_x2 - rect_x1
#     box_h = rect_y2 - rect_y1

#     # 5) read + resize the omega symbol to that box
#     fg_symbol = cv2.imread(omega_symbol_path, cv2.IMREAD_UNCHANGED)
#     if fg_symbol is None:
#         print("Warning: cannot read omega symbol at:", omega_symbol_path)
#         return

#     # Optionally draw a white rectangle behind it
#     # cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (255,255,255), -1)

#     # resize
#     if box_w > 1 and box_h > 1:
#         resized_symbol = cv2.resize(fg_symbol, (box_w, box_h), interpolation=cv2.INTER_AREA)
#         # 6) overlay
#         overlay_image_alpha(img, resized_symbol, rect_x1, rect_y1)
#     else:
#         print("Box too small, skipping overlay")


# # ##############################################
# # # 5) MAIN DEMO
# # ##############################################
# def main():
#     model = EfficientNetB7(weights='imagenet')

#     # Paths
#     image_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\S__11509778_0.jpg'
#     output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\S__11509778_0.jpg'
#     json_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\S__11509778_0.json'
#     omega_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\Greek_lc_omega.png'

#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Could not read image at " + image_path)

#     circle = detect_consistent_circle(image)
#     if circle:
#         x, y, r = circle
#         output = image.copy()

#         # Optional classification
#         roi = output[y-r:y+r, x-r:x+r]
#         roi = cv2.resize(roi, (600, 600))
#         roi_pre = preprocess_input(np.expand_dims(roi, axis=0))
#         preds = model.predict(roi_pre)
#         top_label = decode_predictions(preds, top=1)[0][0][1]
#         confidence = float(preds[0][np.argmax(preds)])
#         result = {
#             "label": top_label,
#             "confidence": confidence,
#             "geometry": {"center": (int(x), int(y)), "radius": int(r)}
#         }
#         with open(json_path, 'w') as f:
#             json.dump(result, f)

#         # scale factors
#         scale_factor = r / 100.0
#         thickness = max(1.5, int(scale_factor * 3))
#         font_scale = max(1.0, scale_factor * 0.5)
#         arrow_tip_len = int(r * 0.2)

#         # 1) Draw main circle
#         cv2.circle(output, (x, y), r, (0, 255, 0), int(thickness))

#         # 2) Radius line + label "r"
#         end_point_r = (x - r, y)
#         cv2.line(output, (x, y), end_point_r, (0, 255, 255), int(thickness))
#         mid_point_r = ((x + end_point_r[0]) // 2, (y + end_point_r[1]) // 2)
#         label_center_r = (
#             mid_point_r[0] + int(15*scale_factor),
#             mid_point_r[1] - int(20*scale_factor)
#         )
#         draw_text_with_background_centered(
#             output, 'r', label_center_r,
#             font_scale=font_scale,
#             color=(0, 0, 255),
#             thickness=int(thickness),
#             bg_color=(255, 255, 255),
#             padding=5
#         )

#         # 3) Arrow line + label "a"
#         angle_a = 45
#         start_point_a = (
#             int(x + r*np.cos(np.deg2rad(angle_a))),
#             int(y - r*np.sin(np.deg2rad(angle_a)))
#         )
#         end_point_a = (
#             int(start_point_a[0] - 0.5*r*np.cos(np.deg2rad(angle_a))),
#             int(start_point_a[1] + 0.5*r*np.sin(np.deg2rad(angle_a)))
#         )
#         cv2.arrowedLine(output, start_point_a, end_point_a, (0,255,255), int(thickness), tipLength=0.3)

#         mid_point_a = ((start_point_a[0]+end_point_a[0])//2, (start_point_a[1]+end_point_a[1])//2)
#         label_center_a = (
#             mid_point_a[0] - int(15*scale_factor),
#             mid_point_a[1] - int(20*scale_factor)
#         )
#         draw_text_with_background_centered(
#             output, 'a', label_center_a,
#             font_scale=font_scale,
#             color=(0,0,255),
#             thickness=int(thickness),
#             padding=5
#         )

#         # 4) Arc for angular velocity => embed PNG resized
#         startAngle = -210
#         endAngle   = -270
#         curved_radius = int(r*0.7)

#         draw_curved_arrow(
#             output,
#             center=(x,y),
#             radius=curved_radius,
#             start_angle=startAngle,
#             end_angle=endAngle,
#             color=(0,255,255),
#             thickness=int(thickness),
#             tip_len=int(r*0.3),
#             font_scale=font_scale,
#             omega_symbol_path=omega_path
#         )

#         # Save
#         cv2.imwrite(output_path, output)
#         print("Done. Result saved to:", output_path)
#     else:
#         print("No circles detected")


# if __name__ == "__main__":
#     main()


import os
import cv2
import numpy as np
import json
from collections import defaultdict
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


#####################################
# 1) UTILITY: Overlays an RGBA image
#####################################
def overlay_image_alpha(bg, fg, x, y):
    """
    Overlay 'fg' onto 'bg' at (x, y).
    If 'fg' has 4 channels => alpha blend each pixel.
    """
    fh, fw = fg.shape[:2]
    for row in range(fh):
        for col in range(fw):
            # Check bounds
            if (y + row >= bg.shape[0]) or (x + col >= bg.shape[1]):
                continue
            alpha = 1.0
            # If 'fg' is RGBA => the 4th channel is alpha
            if fg.shape[2] == 4:
                alpha = fg[row, col, 3] / 255.0
            for c in range(3):  # B, G, R channels
                bg[y + row, x + col, c] = (
                    alpha * fg[row, col, c]
                    + (1 - alpha) * bg[y + row, x + col, c]
                )


##############################################
# 2) CENTERED TEXT: for "r" and "a" labels
##############################################
def draw_text_with_background_centered(
    image,
    text,
    center_pt,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    color=(0, 0, 255),
    thickness=2,
    bg_color=(255, 255, 255),
    padding=10
):
    """
    Draw text so 'center_pt' is the center of the padded white rectangle,
    with text also centered inside that rectangle.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    rect_x1 = int(center_pt[0] - (text_width / 2) - padding)
    rect_y1 = int(center_pt[1] - (text_height / 2) - padding)
    rect_x2 = int(center_pt[0] + (text_width / 2) + padding)
    rect_y2 = int(center_pt[1] + (text_height / 2) + baseline + padding)

    # Draw white rectangle
    cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    text_x = int(center_pt[0] - (text_width / 2))
    text_y = int(center_pt[1] + (text_height / 2))

    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)


##############################################
# 3) DETECT: More robust circle detection
##############################################
def detect_consistent_circle(
    image,
    debug=False,
    iterations=1,
    param2_values=(30, 25, 20, 15),
    canny_thresh1=80,
    canny_thresh2=150,
    morph_kernel_size=5
):
    """
    Attempts circle detection by:
      1) Converting to grayscale
      2) Applying CLAHE for contrast enhancement
      3) Using Canny edge detection
      4) (Optional) Morphological closing to reduce noise
      5) Trying multiple param2 thresholds in HoughCircles
      6) Grouping similar circles and returning the most frequent

    Parameters:
      image (np.ndarray): BGR image
      debug (bool): If True, show intermediate debug windows
      iterations (int): Re-run the detection multiple times
      param2_values (tuple): Different param2 thresholds for HoughCircles
      canny_thresh1, canny_thresh2 (int): Canny edge thresholds
      morph_kernel_size (int): Size for morphological closing kernel

    Returns:
      (cx, cy, r) as the best circle, or None if none detected.
    """
    detected_circles = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    min_dim = min(height, width)

    # 1) Contrast-Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 2) Canny Edge Detection
    edges = cv2.Canny(enhanced_gray, canny_thresh1, canny_thresh2)

    # 3) Optional morphological closing to strengthen circle edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if debug:
        cv2.imshow("1 - Gray", gray)
        cv2.imshow("2 - Enhanced Gray", enhanced_gray)
        cv2.imshow("3 - Canny Edges", edges)
        cv2.imshow("4 - Closed Edges", closed_edges)
        cv2.waitKey(1)

    # Re-run detection multiple times if desired
    for _ in range(iterations):
        # Try multiple param2 thresholds
        for p2 in param2_values:
            dp = 1.4
            minDist = int(min_dim * 0.2)
            minRadius = int(min_dim * 0.05)
            maxRadius = int(min_dim * 0.45)

            circles = cv2.HoughCircles(
                closed_edges,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=minDist,
                param1=100,   # Upper threshold for Canny inside Hough
                param2=p2,    # Accumulator threshold
                minRadius=minRadius,
                maxRadius=maxRadius
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                detected_circles.extend(circles)

    # Group similar circles
    if not detected_circles:
        return None

    circle_groups = defaultdict(list)
    for (cx, cy, rr) in detected_circles:
        # Group by “binned” center & radius
        key = (cx // 30, cy // 30, rr // 10)
        circle_groups[key].append((cx, cy, rr))

    # Pick the group with the most members
    best_group = max(circle_groups.values(), key=len)
    # Average them to get a final circle
    avg_circle = np.mean(best_group, axis=0).astype(int)
    cx, cy, r = avg_circle
    return (int(cx), int(cy), int(r))


##############################################
# 4) ARC + ARROW TIP, Resized Omega Symbol
##############################################
def draw_curved_arrow(
    img,
    center,
    radius,
    start_angle,
    end_angle,
    color=(0, 165, 255),
    thickness=4,
    tip_len=20,
    font_scale=1.0,
    omega_symbol_path=r".\image_raw\Greek_lc_omega.png"
):
    """
    Draw an arc + arrow tips. Then compute a bounding-box as if
    we were drawing text 'dummy' => resize 'omega_symbol.png'
    to exactly fill that rectangle, and overlay it.
    """

    # 1) Arc
    cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

    # 2) Arrow tip lines
    end_rad = np.deg2rad(end_angle)
    end_x = center[0] + radius * np.cos(end_rad)
    end_y = center[1] + radius * np.sin(end_rad)

    tangent_angle = end_rad + np.deg2rad(90)
    spread = np.deg2rad(30)
    tip1 = (
        end_x + tip_len * np.cos(tangent_angle + spread),
        end_y + tip_len * np.sin(tangent_angle + spread)
    )
    tip2 = (
        end_x + tip_len * np.cos(tangent_angle - spread),
        end_y + tip_len * np.sin(tangent_angle - spread)
    )

    cv2.line(img, (int(end_x), int(end_y)), (int(tip1[0]), int(tip1[1])), color, thickness)
    cv2.line(img, (int(end_x), int(end_y)), (int(tip2[0]), int(tip2[1])), color, thickness)

    # 3) Decide where to place the bounding box => arc midpoint
    mid_angle = (start_angle + end_angle) / 2.0
    mid_rad = np.deg2rad(mid_angle)
    mid_x = int(center[0] + radius * np.cos(mid_rad))
    mid_y = int(center[1] + radius * np.sin(mid_rad))

    # We'll offset it upward a bit
    offset = 0.2 * radius
    rect_center_x = mid_x
    rect_center_y = int(mid_y - offset - 30)

    # 4) Let's measure a bounding box as if text = "x"
    text = "x"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # compute rectangle corners around rect_center
    padding = 10
    rect_x1 = int(rect_center_x - (text_width / 2) - padding)
    rect_y1 = int(rect_center_y - (text_height / 2) - padding)
    rect_x2 = int(rect_center_x + (text_width / 2) + padding)
    rect_y2 = int(rect_center_y + (text_height / 2) + baseline + padding)

    # compute final bounding box size
    box_w = rect_x2 - rect_x1
    box_h = rect_y2 - rect_y1

    # 5) read + resize the omega symbol to that box
    fg_symbol = cv2.imread(omega_symbol_path, cv2.IMREAD_UNCHANGED)
    if fg_symbol is None:
        print("Warning: cannot read omega symbol at:", omega_symbol_path)
        return

    # resize
    if box_w > 1 and box_h > 1:
        resized_symbol = cv2.resize(fg_symbol, (box_w, box_h), interpolation=cv2.INTER_AREA)
        # 6) overlay
        overlay_image_alpha(img, resized_symbol, rect_x1, rect_y1)
    else:
        print("Box too small, skipping overlay")


##############################################
# 5) MAIN DEMO
##############################################
def main():
    model = EfficientNetB7(weights='imagenet')

    # Paths
    image_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\S__129433700.jpg'
    output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\S__129433700.jpg'
    json_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\S__129433700.json'
    omega_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\Greek_lc_omega.png'

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image at " + image_path)

    # Attempt circle detection
    circle = detect_consistent_circle(
        image,
        debug=False,         # set True to see intermediate windows
        iterations=1,        # you can increase this to 2 or 3 if needed
        param2_values=(30, 25, 20, 15),
        canny_thresh1=80,
        canny_thresh2=150,
        morph_kernel_size=5
    )

    if circle:
        x, y, r = circle
        output = image.copy()

        # Optional classification (ROI around circle)
        roi = output[y-r:y+r, x-r:x+r]
        if roi.size > 0:
            roi = cv2.resize(roi, (600, 600))
            roi_pre = preprocess_input(np.expand_dims(roi, axis=0))
            preds = model.predict(roi_pre)
            top_label = decode_predictions(preds, top=1)[0][0][1]
            confidence = float(preds[0][np.argmax(preds)])
            result = {
                "label": top_label,
                "confidence": confidence,
                "geometry": {"center": (int(x), int(y)), "radius": int(r)}
            }
        else:
            # If the circle is partially out of bounds
            result = {
                "label": "unknown",
                "confidence": 0.0,
                "geometry": {"center": (int(x), int(y)), "radius": int(r)}
            }

        # Write JSON
        with open(json_path, 'w') as f:
            json.dump(result, f)

        # scale factors
        scale_factor = r / 100.0
        thickness = max(1.5, int(scale_factor * 3))
        font_scale = max(1.0, scale_factor * 0.5)
        arrow_tip_len = int(r * 0.2)

        # 1) Draw main circle
        cv2.circle(output, (x, y), r, (0, 255, 0), int(thickness))

        # 2) Radius line + label "r"
        end_point_r = (x - r, y)
        cv2.line(output, (x, y), end_point_r, (0, 0, 0), int(thickness))
        mid_point_r = ((x + end_point_r[0]) // 2, (y + end_point_r[1]) // 2)
        label_center_r = (
            mid_point_r[0] + int(15 * scale_factor),
            mid_point_r[1] - int(20 * scale_factor)
        )
        draw_text_with_background_centered(
            output, 'r', label_center_r,
            font_scale=font_scale,
            color=(0, 0, 255),
            thickness=int(thickness),
            bg_color=(255, 255, 255),
            padding=5
        )

        # 3) Arrow line + label "a"
        angle_a = 45
        start_point_a = (
            int(x + r * np.cos(np.deg2rad(angle_a))),
            int(y - r * np.sin(np.deg2rad(angle_a)))
        )
        end_point_a = (
            int(start_point_a[0] - 0.5 * r * np.cos(np.deg2rad(angle_a))),
            int(start_point_a[1] + 0.5 * r * np.sin(np.deg2rad(angle_a)))
        )
        cv2.arrowedLine(output, start_point_a, end_point_a, (0, 0, 0), int(thickness), tipLength=0.3)

        mid_point_a = (
            (start_point_a[0] + end_point_a[0]) // 2,
            (start_point_a[1] + end_point_a[1]) // 2
        )
        label_center_a = (
            mid_point_a[0] - int(15 * scale_factor),
            mid_point_a[1] - int(20 * scale_factor)
        )
        draw_text_with_background_centered(
            output, 'a', label_center_a,
            font_scale=font_scale,
            color=(0, 0, 255),
            thickness=int(thickness),
            padding=5
        )

        # 4) Arc for angular velocity => embed PNG resized
        startAngle = -210
        endAngle = -270
        curved_radius = int(r * 0.7)

        draw_curved_arrow(
            output,
            center=(x, y),
            radius=curved_radius,
            start_angle=startAngle,
            end_angle=endAngle,
            color=(0, 0, 0),
            thickness=int(thickness),
            tip_len=int(r * 0.3),
            font_scale=font_scale,
            omega_symbol_path=omega_path
        )

        # Save
        cv2.imwrite(output_path, output)
        print("Done. Result saved to:", output_path)
    else:
        print("No circles detected")


if __name__ == "__main__":
    main()



# ###############
# ###############
# ###############
# def detect_multiple_circles(image, iterations=10):
#     """
#     Detect multiple circles and return all detected circles.
#     """
#     detected_circles = []
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     height, width = gray.shape
#     min_dim = min(height, width)

#     # Contrast Limited Adaptive Histogram Equalization
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#     for _ in range(iterations):
#         enhanced_gray = clahe.apply(gray)
#         kernel_size = int(min_dim * 0.02) // 2 * 2 + 1
#         blurred = cv2.GaussianBlur(enhanced_gray, (kernel_size, kernel_size), 0)

#         _ = cv2.adaptiveThreshold(
#             blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY, 11, 2
#         )

#         params = {
#             'dp': 1.6,
#             'minDist': int(min_dim * 0.2),
#             'param1': 100,
#             'param2': 30,
#             'minRadius': int(min_dim * 0.05),
#             'maxRadius': int(min_dim * 0.45)
#         }
#         circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **params)
#         if circles is not None:
#             circles = np.round(circles[0, :]).astype("int")
#             detected_circles.extend(circles)

#     return detected_circles



# ##############################################
# # 5) MAIN DEMO
# ##############################################
# def main():
#     model = EfficientNetB7(weights='imagenet')

#     # Paths
#     image_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\fan22.png'
#     output_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\fan22.png'
#     json_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\fan22.json'
#     omega_path = r'D:\code2\update 4\backend\fastapi-gpt\image_raw\Greek_lc_omega.png'

#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Could not read image at " + image_path)

#     circles = detect_multiple_circles(image)
#     if circles:
#         for circle in circles:
#             x, y, r = circle
#             output = image.copy()

#             # Check if the detected circle is within the image bounds
#             if y - r >= 0 and y + r <= image.shape[0] and x - r >= 0 and x + r <= image.shape[1]:
#                 # Extract the region of interest (ROI)
#                 roi = output[y-r:y+r, x-r:x+r]

#                 # Check if the ROI is valid (non-empty)
#                 if roi.size != 0:
#                     # Resize the ROI to the desired size, e.g. resize to 224x224
#                     custom_size = (600, 600)  # Replace with your preferred size
#                     try:
#                         # Resize the ROI to the desired size
#                         roi = cv2.resize(roi, custom_size)  # Resize ROI to custom size
#                         roi_pre = preprocess_input(np.expand_dims(roi, axis=0))
#                         preds = model.predict(roi_pre)
#                         top_label = decode_predictions(preds, top=1)[0][0][1]
#                         confidence = float(preds[0][np.argmax(preds)])
#                         result = {
#                             "label": top_label,
#                             "confidence": confidence,
#                             "geometry": {"center": (int(x), int(y)), "radius": int(r)}
#                         }

#                         # Save the result as JSON
#                         json_path = r'D:\code2\update 4\backend\fastapi-gpt\image_result\fan11.json'
#                         with open(json_path, 'w') as f:
#                             json.dump(result, f)

#                         # Perform the other image operations (circle drawing, labels, etc.)
#                         scale_factor = r / 100.0
#                         thickness = max(1.5, int(scale_factor * 3))
#                         font_scale = max(1.0, scale_factor * 0.5)
#                         arrow_tip_len = int(r * 0.2)

#                         # Draw main circle
#                         cv2.circle(output, (x, y), r, (0, 255, 0), int(thickness))

#                         # 2) Radius line + label "r"
#                         end_point_r = (x - r, y)
#                         cv2.line(output, (x, y), end_point_r, (0, 255, 255), int(thickness))
#                         mid_point_r = ((x + end_point_r[0]) // 2, (y + end_point_r[1]) // 2)
#                         label_center_r = (
#                             mid_point_r[0] + int(15 * scale_factor),
#                             mid_point_r[1] - int(20 * scale_factor)
#                         )
#                         draw_text_with_background_centered(
#                             output, 'r', label_center_r,
#                             font_scale=font_scale,
#                             color=(0, 0, 255),
#                             thickness=int(thickness),
#                             bg_color=(255, 255, 255),
#                             padding=5
#                         )

#                         # 3) Arrow line + label "a"
#                         angle_a = 45
#                         start_point_a = (
#                             int(x + r * np.cos(np.deg2rad(angle_a))),
#                             int(y - r * np.sin(np.deg2rad(angle_a)))
#                         )
#                         end_point_a = (
#                             int(start_point_a[0] - 0.5 * r * np.cos(np.deg2rad(angle_a))),
#                             int(start_point_a[1] + 0.5 * r * np.sin(np.deg2rad(angle_a)))
#                         )
#                         cv2.arrowedLine(output, start_point_a, end_point_a, (0, 255, 255), int(thickness), tipLength=0.3)

#                         mid_point_a = ((start_point_a[0] + end_point_a[0]) // 2, (start_point_a[1] + end_point_a[1]) // 2)
#                         label_center_a = (
#                             mid_point_a[0] - int(15 * scale_factor),
#                             mid_point_a[1] - int(20 * scale_factor)
#                         )
#                         draw_text_with_background_centered(
#                             output, 'a', label_center_a,
#                             font_scale=font_scale,
#                             color=(0, 0, 255),
#                             thickness=int(thickness),
#                             padding=5
#                         )

#                         # 4) Arc for angular velocity => embed PNG resized
#                         startAngle = -210
#                         endAngle = -270
#                         curved_radius = int(r * 0.7)

#                         draw_curved_arrow(
#                             output,
#                             center=(x, y),
#                             radius=curved_radius,
#                             start_angle=startAngle,
#                             end_angle=endAngle,
#                             color=(0, 255, 255),
#                             thickness=int(thickness),
#                             tip_len=int(r * 0.3),
#                             font_scale=font_scale,
#                             omega_symbol_path=omega_path
#                         )

#                         # Save the output image
#                         cv2.imwrite(output_path, output)
#                         print("Done. Result saved to:", output_path)
#                     except cv2.error as e:
#                         print(f"Error during resizing: {e}")
#                 else:
#                     print("Detected circle ROI is empty, skipping.")
#             else:
#                 print(f"Circle at ({x}, {y}, {r}) is out of bounds, skipping.")
#     else:
#         print("No circles detected")

# if __name__ == "__main__":
#     main()