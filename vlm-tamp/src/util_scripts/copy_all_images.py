import os
import shutil
dataset  = "../dalle_4/"
for task in os.listdir(dataset):
    os.mkdir(dataset+task+'/'+"all_images")
    target_path = dataset+task+'/all_images'

    for action in os.listdir(dataset+task):
        if action == "all_images":
            break
        for f in os.listdir(dataset+task+'/'+action+'/0/'):
            shutil.copy(dataset+task+'/'+action+'/0/'+f, target_path)
        for f in os.listdir(dataset+task+'/'+action+'/2/'):
            shutil.copy(dataset+task+'/'+action+'/2/'+f, target_path)



