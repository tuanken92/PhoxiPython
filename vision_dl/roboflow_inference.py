from roboflow import Roboflow
import time
from datetime import date
rf = Roboflow(api_key="ovlwhtl9Yk3aB8uI6wSc")
project = rf.workspace().project("box-segmentation-1zcsw")
model = project.version(2).model


def current_milli_time():
    return round(time.time() * 1000)

def print_timestamp():
    print("time stamp: {0}".format(round(time.time() * 1000)))


img = "test1.png"

print_timestamp()
t1 = current_milli_time()
# infer on a local image
print(model.predict(img).json())
print_timestamp()
t2 = current_milli_time() - t1
print("Time processing = {0}".format(t2))

# infer on an image hosted elsewhere
#print(model.predict("URL_OF_YOUR_IMAGE").json())

# save an image annotated with your predictions
model.predict(img).save("prediction.jpg")