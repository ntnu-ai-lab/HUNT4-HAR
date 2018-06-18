import matplotlib.pyplot as plt
import PIL.Image as Image
import glob
import numpy as np

image_folder = "imgs/=3=3_units"
fig = plt.figure(figsize=(20, 10))
n_units = image_folder.split("imgs/",1)[1]
all_imgs = glob.glob(image_folder+"/*")
combined_image = np.zeros(shape=(720,1440,3))
iterator = 0
for img in all_imgs:
    image = np.asarray(Image.open(img))
    combined_image+=image
    iterator+=1

combined_image = (combined_image//iterator).astype(np.uint8)
plt.imshow(combined_image)
plt.savefig("imgs/avg_images/"+n_units+".pdf")