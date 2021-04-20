import cv2
import numpy as np


class ProcessImage():
    def __init__(self):
        image_location = "../images/20210211_140059669_iOS.jpg"
        self.image = self.read_image(image_location)
        self.lower_green = np.array([65,50,50])
        self.upper_green = np.array([87,255,255])
        self.kernel = np.ones((5,5),np.uint8)


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

    def main(self):
        image = self.resize_image(self.image)
        self.show_image(image)
        image = self.change_to_hsv(image)
        image = self.get_binary_image(image, self.lower_green, self.upper_green)
        self.show_image(image)
        image = self.erode_image(image, self.kernel)
        #self.show_image(image)
        image = self.dilate_image(image, self.kernel)
        self.show_image(image)

if __name__ == "__main__":
    process_image = ProcessImage()
    process_image.main()
