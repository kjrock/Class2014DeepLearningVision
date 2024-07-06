import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Video(animation.FuncAnimation):
    def __init__(self, device=0, fig=None, frames=None, interval=50, repeat_dealy=5):
        if fig is None:
            self.fig = plt.figure()
            plt.axis('off')

        super().__init__(self.fig, self.updateFrame, init_func=self.init,
                         frames=frames, interval=interval, repeat_delay=repeat_dealy)
        self.cap = cv2.VideoCapture(device)
        print('start capture')


    def init(self):
        ret, self.frame = self.cap.read()
        if ret:
            self.im = plt.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        return self.im

    def updateFrame(self, k):
        ret, self.frame = self.cap.read()
        if ret:
            self.im.set_array(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
        print('finish!!!!')

camear = Video()
plt.show()
camear.close()
