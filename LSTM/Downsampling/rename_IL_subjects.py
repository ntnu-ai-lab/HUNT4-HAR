import os


path_to_downsamples = "/PATH/TO/Downsampled-data"
downsamples = ["EVERY_OTHER", "TWO_AND_TWO_AVG", "RESAMPLE", "RESAMPLE_POLY", "DECIMATE"]


for i in range(len(downsamples)):
    path = path_to_downsamples+"/"+downsamples[i]+"/"
    for file_name in os.listdir(path):
        for file in os.listdir(path+file_name):
            if file[4:].startswith("RT_UT"):
                orig_name = file[0:4]
                new_name = path+file_name+"/"+orig_name+"Axivity_THIGH_Right.csv"
                old_path = path + file_name + "/"+file
                os.rename(old_path, new_name)
            elif file[4:].startswith("LT_UT"):
                orig_name = file[0:4]
                new_name = path+file_name+"/"+orig_name+"Axivity_THIGH_Left.csv"
                old_path = path + file_name + "/"+file
                os.rename(old_path, new_name)
            elif file[4:].startswith("LB"):
                orig_name = file[0:4]
                new_name = path+file_name+"/"+orig_name+"Axivity_BACK_back.csv"
                old_path = path + file_name + "/"+file
                os.rename(old_path, new_name)
            elif file[4:].startswith("labels"):
                orig_name = file[0:4]
                new_name = path+file_name+"/"+orig_name + "GoPro_LAB_All.csv"
                old_path = path + file_name + "/"+file
                os.rename(old_path, new_name)
