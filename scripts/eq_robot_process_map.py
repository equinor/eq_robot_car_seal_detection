import cv2
import numpy as np

class ProcessMap():
    def __init__(self):
        image_location = "../images/localization-inspector.pgm"
        self.image = self.read_image(image_location)


    def read_image(self, image_location):
        return cv2.imread(image_location, 0)

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
        self.show_image(self.image)

if __name__ == "__main__":
    process_map = ProcessMap()
    process_map.main()