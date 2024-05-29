import os
import shutil
import pandas as pd
 
csv_file = pd.read_csv("./data/iemocap_new.csv")
files = csv_file["FileName"]

# root = "./IEMOCAP_full_release/"
root  ="./IEMOCAP/"
destination = "./data/iemocap/"
os.makedirs(destination, exist_ok=True)

for file in files:
    session = file[4:5]
    session = "Session" + str(session)
    src = root + session + "/sentences/wav/" + "_".join((file.split(".")[0].split("_")[:-1])) + "/" + file + ".wav"
    dst = destination + file + ".wav"
    shutil.copyfile(src, dst)

    