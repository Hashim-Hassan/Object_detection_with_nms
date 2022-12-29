from Detector import *

# Model 1 : SSD MobileNet V2 FPNLite 320*320
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

# Model 2 : EfficientDet D4 1024*1024
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

classfile = 'coco.names'
imagePath = "test/2.jpg"

detector = Detector()
detector.readClasses(classfile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath)