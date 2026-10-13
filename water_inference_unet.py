import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os
import cv2

MASK_THRESHOLD = 0.5
class u_net_model:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Input details: {self.input_details}")
        print(f"Output details: {self.output_details}")
        self.input_scale, self.input_zero = self.input_details[0]['quantization']
        self.output_scale, self.output_zero = self.output_details[0]['quantization']
        _, self.input_h, self.input_w, self.input_c = self.input_details[0]['shape']
        self.dtype = self.input_details[0]['dtype']

    def load_image(self, image_path):
        img = Image.open(image_path)

        # Ensure RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to model input size
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)

        # Convert to numpy
        img_np = np.array(img, dtype=np.float32)

        # Quantize image to int8
        # q = f/scale + zero_point
        img_q = img_np / 255.0               # normalize [0,1]
        
        
        if self.dtype == np.int8:
            img_q = img_q / self.input_scale + self.input_zero
            img_q = np.clip(img_q, -128, 127).astype(np.int8)            
            # Add batch dimension
        input_tensor = np.expand_dims(img_q, axis=0)
        return input_tensor
    
    
    def save_mask(self, output_f, save_path, threshold=MASK_THRESHOLD):
        mask = (output_f > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_img.save(save_path)
        
    def calculate_percentage(self, output_f, threshold=MASK_THRESHOLD):
        mask = (output_f > threshold).astype(np.uint8).flatten()
        water_pixels = np.sum(mask)
        total_pixels = len(mask)
        percentage = (water_pixels / total_pixels) * 100
        return percentage
    
    
    def predict(self, image_path):
        
        t = time.time()
        input_tensor = self.load_image(image_path)
        
        t1 = time.time()
        print(f"Preprocessing time: {t1 - t:.3f} seconds")
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)

        t2 = time.time()
        print(f"Set tensor time: {t2 - t1:.3f} seconds")
        # Run inference
        self.interpreter.invoke()

        t3 = time.time()
        print(f"Inference time: {t3 - t2:.3f} seconds")
        
        # Retrieve output
        # Dequantize: f = (q - zero_point) * scale
        if self.dtype == np.int8:
            output_q = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # shape: 256x256x1
            output_f = (output_q.astype(np.float32) - self.output_zero) * self.output_scale
        else:
            output_f = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        output_f = np.squeeze(output_f) 
        
        t4 = time.time()
        print(f"Postprocessing time: {t4 - t3:.3f} seconds")
        
        print(f"Total time: {t4 - t:.3f} seconds")
        return output_f

# class of functions used for anaylzing the image for aruco markers and measuring distances
class aruco_vision: 

    def clean_mask(mask, kernel=np.ones((3,3),np.uint8)):
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cleaned_mask

    def extract_high_boundary(mask, x_coord, points=5 ):
        
        y_indices, x_indices = np.where(mask > 0)
        y_aruco = y_indices[x_indices == x_coord]
        if len(y_aruco) >= points:
            top_y = np.sort(y_aruco)[:points]
            return top_y
        return []
    
    def detect_aruco_coords(image_path, dict=cv2.aruco.DICT_4X4_100, output_w=256, output_h=256):
        image = cv2.imread(image_path)

        aruco_dict = cv2.aruco.getPredefinedDictionary(dict)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, rejected = detector.detectMarkers(image)

        if ids is not None:
            print(f"Detected {len(ids)} marker(s):\n")
            
            # Draw detected marker boundaries and IDs on the image
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

            for marker_id, corner_set in zip(ids.flatten(), corners):
                # Reshape corner array to 4 (x, y) points
                pts = corner_set[0]
                top_left, top_right, bottom_right, bottom_left = pts

                # Calculate center coordinate (centroid)
                center_x = int(pts[:, 0].mean())
                center_y = int(pts[:, 1].mean())

                print(f"--- Marker ID: {marker_id} ---")
                print(f" Top-Left:     {pts[0]}")
                print(f" Top-Right:    {pts[1]}")
                print(f" Bottom-Right: {pts[2]}")
                print(f" Bottom-Left:  {pts[3]}")
                print(f" Center (X, Y): ({center_x}, {center_y})\n")

                # Draw a dot at the center point
                cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        else:
            print("No ArUco markers found in the image.")
            return

        # cv2.imshow("ArUco Detection", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return center_x,center_y

    def aruco_map_to_output(image_path, x, y, output_w, output_h):
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size  
        x_out = int(round(x * (output_w/orig_w)))
        y_out = int(round(y * (output_h/orig_h)))

        return x_out, y_out

    def draw_result(image_path, aruco_x, aruco_y, water_y):
        image = cv2.imread(image_path)
        cv2.circle(image, (aruco_x, aruco_y), 5, (0, 0, 255), -1)
        cv2.line(image, (aruco_x, aruco_y), (aruco_x, water_y), (255, 255, 0), 1)
        diff_y = water_y - aruco_y
        text = f"{diff_y}"
        mid_y = (aruco_y + water_y) // 2
        text_pos = (aruco_x + 10, mid_y)
        cv2.putText(image, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
       
        cv2.imshow("ArUco Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    



    
def main(model_type="8"):

    ### Segmentation processing ###
    if model_type == "8":
        model_path = "models/tiny_unet_int8.tflite"
    else:
        model_path = "models/tiny_unet_fp32.tflite"
    model = u_net_model(model_path)
    
    image_path = 'water_720.png'
    
    if os.path.exists("./photos"):
        image_files = sorted(os.listdir("./photos"))
        if len(image_files) > 0:
            image_path = os.path.join("./photos", image_files[-1])  # Use the latest image
    
    output = model.predict(image_path)
    percentage = model.calculate_percentage(output)
    import random
    if random.random() < 0.9 and image_path != 'water_720.png':
        os.system('rm -f ' + image_path)
    print(f"Water coverage percentage: {percentage:.2f}%")


    ### Aruco processing ###
    x,y = aruco_vision.detect_aruco_coords("water_720_aruco.png")
    aruco_x,aruco_y = aruco_vision.aruco_map_to_output("water_720_aruco.png", x,y,256,256)
    print(f"aruco coordinates in output space is: {aruco_x,aruco_y}")
    
    cleaned_output = aruco_vision.clean_mask(output)
    top_y = aruco_vision.extract_high_boundary(cleaned_output, aruco_x)
    print(f"Top points of the mask are: {top_y}")
    print(f"Distance in output space of U_net between center of Aruco marker and water surface is: {top_y[0] - aruco_y}")
    model.save_mask(output, 'test/image.png')
    aruco_vision.draw_result('test/image.png',aruco_x, aruco_y, top_y[0])
    
    return int(percentage)


    
if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "8"
    main(arg)
