import cv2
import numpy as np

class Cartoonify:
    def __init__(self, image):
        self.image = cv2.imread(image)

    def get_clusters(self, data):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        ret, label, center = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result=center[label.flatten()]
        return result

    def apply_filter(self):
        originalImageResized = cv2.resize(self.image, (640,480))
        data = np.float32(originalImageResized).reshape((-1,3))
        result = self.get_clusters(data)
        result = result.reshape(originalImageResized.shape)  
        colorImage = cv2.bilateralFilter(result, 5, 700, 700)     
        filtered_image = cv2.resize(colorImage, (480,520))
        return filtered_image

    def original_image(self):
       return self.image

if __name__=='__main__':
    IMAGE = "cartoonify/utils/image.jpg"
    cartoonFilter = Cartoonify(IMAGE)
    original_image = cartoonFilter.original_image()
    cartooned_image = cartoonFilter.apply_filter()
    cv2.imshow('Output', cv2.resize(original_image, (480,520)))
    cv2.waitKey(5000)
    cv2.imshow('Output',cartooned_image)
    cv2.waitKey(5000)