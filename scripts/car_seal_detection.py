import cv2
import numpy as np


class CarSealDetector():
    def __init__(self):
        image_location = "../images/Car seal close ventil nede ved dorken.jpg"
        #image_location = "../images/Carseal open nede ved dork.jpg"
        self.video_location = "../videos/VID_20191008_095527.mp4"
        self.video = False
        self.image = self.read_image(image_location)
        self.lower_green = np.array([65,50,50])
        self.upper_green = np.array([87,255,255])
        self.lower_red_1 = np.array([177,50,50])
        self.upper_red_1 = np.array([180,255,255])
        self.lower_red_2 = np.array([0,50,50])
        self.upper_red_2 = np.array([8,255,255])
        self.kernel = np.ones((3,3),np.uint8)
        self.min_area = 500
        self.max_area = 200000
        self.color = {"red":[0, 0, 255], "green":[0, 255, 0]}


    def read_image(self, image_location):
        return cv2.imread(image_location, cv2.IMREAD_COLOR)

    def change_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def get_binary_image(self, image, lower, upper):
        return cv2.inRange(image, lower, upper)

    def erode_image(self, image, kernel):
        return cv2.erode(image,kernel,iterations = 1)

    def dilate_image(self, image, kernel):
        return cv2.dilate(image,kernel,iterations = 1)

    def resize_image(self, image):
        height = image.shape[0]
        width = image.shape[1]
        scaling_factor = 0.25
        new_width = int(scaling_factor*width)
        new_heigth = int(scaling_factor*height)
        return cv2.resize(image, (new_width, new_heigth))

    def show_image(self, image):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        
        while True:
            k = cv2.waitKey(0)
            if k == 27:
                break
            else:
                continue

        cv2.destroyAllWindows()

    def get_green_mask(self, image):
        return self.get_binary_image(image, self.lower_green, self.upper_green)

    def get_red_mask(self, image):
        mask_1 = self.get_binary_image(image, self.lower_red_1, self.upper_red_1)
        mask_2 = self.get_binary_image(image, self.lower_red_2, self.upper_red_2)
        return mask_1+mask_2

    def get_contours(self, mask):
        try:
            _, cont, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_area = []
            biggest_contour = []
            biggest_area = self.min_area
            for con in cont:
                area = cv2.contourArea(con)
                if self.min_area < area < self.max_area:
                    contours_area.append(con)
                    if area > biggest_area:
                        biggest_area = area
                        biggest_contour = [con]
            return biggest_contour
        except(ValueError, ZeroDivisionError):
            return []

    def display_bounding_box(self, contours, image, color):
        for con in contours:
            M = cv2.moments(con)
            center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            x,y,w,h = cv2.boundingRect(con)

            cv2.rectangle(image,(x,y),(x+w,y+h),self.color[color])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.8
            cv2.putText(image, color+' ziptie',(x,y),font,font_size,self.color[color],2,cv2.LINE_AA)

    def check_for_ziptie(self, image):
            image_hsv = self.change_to_hsv(image)
            green_mask = self.get_green_mask(image_hsv)
            green_mask = self.erode_image(green_mask, self.kernel)
            #self.show_image(image)
            green_mask = self.dilate_image(green_mask, self.kernel)

            red_mask = self.get_red_mask(image_hsv)
            red_mask = self.erode_image(red_mask, self.kernel)
            red_mask = self.dilate_image(red_mask, self.kernel)


            red_contours = self.get_contours(red_mask)
            if len(red_contours) > 0:
                a=1
                self.display_bounding_box(red_contours, image, "red")

            green_contours = self.get_contours(green_mask)
            if len(green_contours) > 0:
                self.display_bounding_box(green_contours, image, "green")
            return image

    def main(self):
        if self.video:
            cap = cv2.VideoCapture(self.video_location)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter('../videos/outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.flip(frame, 0)
                frame = cv2.flip(frame, 1)
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                
                frame = self.check_for_ziptie(frame)
                out.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows()
        else:
            #image = self.resize_image(self.image)
            image = self.image
            image = self.check_for_ziptie(image)
            cv2.imwrite('../images/output.jpg', image)
            self.show_image(image)


if __name__ == "__main__":
    car_seal_detector = CarSealDetector()
    car_seal_detector.main()
