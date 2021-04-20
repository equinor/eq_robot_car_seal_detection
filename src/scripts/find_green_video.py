import cv2
import numpy as np

class ProcessVideo():
    def __init__(self):
        video_location = "../videos/VID_20191008_095527.mp4"
        self.cap = self.open_cap(video_location)
        self.lower_green = np.array([70,80,80])
        self.upper_green = np.array([87,255,255])
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        self.out = cv2.VideoWriter('../videos/outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    def open_cap(self, video_location):
        return cv2.VideoCapture(video_location)

    def change_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def get_binary_image(self, image, lower, upper):
        return cv2.inRange(image, lower, upper)


    def main(self):
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_hsv = self.change_to_hsv(frame)
            mask = self.get_binary_image(frame_hsv, self.lower_green, self.upper_green)
            frame = cv2.bitwise_and(frame,frame, mask= mask)
            self.out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video = ProcessVideo()
    process_video.main()
