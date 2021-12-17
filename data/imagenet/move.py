import os
import shutil

def move_file():
    shutil.copyfile("imagenet-object-localization-challenge19.tar.gz", "/content/imagenet-object-localization-challenge19.tar.gz")
    cmd = "tar -xf '/content/imagenet-object-localization-challenge19.tar.gz'"
    os.system(cmd)