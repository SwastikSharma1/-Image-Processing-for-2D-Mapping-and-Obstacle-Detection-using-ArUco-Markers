import cv2
import numpy as np

class Arena:
    def __init__(self, image_path):
        self.image_path = image_path
        self.obstacles = 0  
        self.aruco_ids = []  
        self.total_area = 0  
        self.scale = 1.25  # Scale for area calculation
        
        # Predefined ArUco dictionaries
        self.aruco_dicts = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        }

    def detect_obstacles(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        lower_gray = 110
        upper_gray = 190
        mask = cv2.inRange(blurred, lower_gray, upper_gray)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self.total_area = 0  
        obstacle_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2200:  # Threshold to avoid small artifacts
                epsilon = 0.0001 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                obstacle_contours.append(approx_contour)
                self.total_area += float(area)  # Sum the area of detected contours
        
        self.obstacles = len(obstacle_contours)
        return obstacle_contours

    def detect_aruco_markers(self, frame):
        self.aruco_ids = []
        best_detection = (None, None, 0)
        
        for dictionary in self.aruco_dicts.values():
            aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            
            corners, ids, _ = detector.detectMarkers(frame)
            
            if ids is not None and len(ids) > best_detection[2]:
                best_detection = (corners, ids, len(ids))
        
        if best_detection[1] is not None:
            self.aruco_ids = best_detection[1].flatten().tolist()
            frame = cv2.aruco.drawDetectedMarkers(frame, best_detection[0], best_detection[1])
            return best_detection[0], best_detection[1], frame
        
        return None, None, frame

    def apply_perspective_transform(self, frame, corners, ids, obstacle_contours):
        if len(corners) >= 4:
            height, width = frame.shape[:2]
            sorted_indices = np.argsort(ids.flatten())
            sorted_corners = [corners[i][0] for i in sorted_indices]
            
            tl = sorted_corners[0][0]
            tr = sorted_corners[1][1]
            br = sorted_corners[2][2]
            bl = sorted_corners[3][3]

            dst = np.array([ 
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]], dtype="float32")

            rect = np.array([tl, tr, br, bl], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
            warped = cv2.warpPerspective(frame, M, (width, height))

            for contour in obstacle_contours:
                transformed_contour = cv2.perspectiveTransform(contour.astype(np.float32).reshape(-1, 1, 2), M)
                cv2.drawContours(warped, [transformed_contour.astype(int)], -1, (0, 255, 0), 3)

            return warped
        return frame

    def calculate_real_area(self, pixel_area):
        real_area = pixel_area * (self.scale ** 2)
        return round(real_area, 2)

    def print_aruco_locations(self, corners, ids, frame):
        if ids is not None:
            for corner, aruco_id in zip(corners, ids.flatten()):
                center_x = int(np.mean(corner[0][:, 0]))
                center_y = int(np.mean(corner[0][:, 1]))
                position = self.get_position(center_x, center_y, frame)
                print(f"Aruco ID: {aruco_id}, Location: {position}")

    def get_position(self, center_x, center_y, frame):
        # Determine position based on coordinates
        if center_x < frame.shape[1] // 2 and center_y < frame.shape[0] // 2:
            return 'Top Left'
        elif center_x >= frame.shape[1] // 2 and center_y < frame.shape[0] // 2:
            return 'Top Right'
        elif center_x >= frame.shape[1] // 2 and center_y >= frame.shape[0] // 2:
            return 'Bottom Right'
        else:
            return 'Bottom Left'

    def process_image(self):
        frame = cv2.imread(self.image_path)
        if frame is not None:
            height, width = frame.shape[:2]
            print(f"Image Width: {width} pixels, Height: {height} pixels")

            # Resize if necessary
            max_width = 800  
            max_height = 800  
            
            if width > max_width or height > max_height:
                scaling_factor = min(max_width / width, max_height / height)
                new_width = int(width * scaling_factor)
                new_height = int(height * scaling_factor)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Step 1: Detect ArUco markers
            corners, ids, frame_with_markers = self.detect_aruco_markers(frame)

            # Step 2: Apply perspective transformation if enough markers are found
            if ids is not None and len(ids) >= 4:
                warped_frame = self.apply_perspective_transform(frame_with_markers, corners, ids, [])

                # Draw contours on the warped frame
                obstacle_contours = self.detect_obstacles(warped_frame)
                for contour in obstacle_contours:
                    cv2.drawContours(warped_frame, [contour], -1, (0, 255, 0), 3)

                cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)  
                cv2.imshow('Processed Image', warped_frame)  

                # Print the ArUco IDs and their locations
                self.print_aruco_locations(corners, ids, frame)

                # Step 4: Calculate total area
                total_area_in_pixels = self.total_area
                real_area = self.calculate_real_area(total_area_in_pixels)
                print(f"Total Area of Detected Obstacles: {total_area_in_pixels} pixels")
                print(f"Estimated Real Area of Obstacles: {real_area} square units")
                print(f"Number of Obstacles Detected: {self.obstacles}")  # Print the number of obstacles

            else:
                print("Not enough ArUco markers detected.")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error loading the image.")

# Example usage
if __name__ == "__main__":
    arena = Arena('new_eyrc.jpg')
    arena.process_image()
