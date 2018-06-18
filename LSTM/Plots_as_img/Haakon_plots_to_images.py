from wand.image import Image
import glob
import os

all_files = glob.glob("/PATH/TO/Recurrent_ANN/plots/*.pdf")

for filename in all_files:
    n_units = filename.split("Bi",1)[1]
    n_units = n_units.split("-19",1)[0]+"_units"
    with Image(filename=filename) as img:
        store_path = "imgs/"+ n_units
        if not os.path.isdir(store_path):
            os.mkdir(store_path)
        save_name = filename.split("/plots/",1)[1][:-4]
        img.convert('jpeg')
        img.save(filename="imgs/"+n_units+"/"+save_name+".jpg")
