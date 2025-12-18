import os
import openai
import cv2
import urllib.request
openai.api_key = os.getenv("OPENAI_API_KEY", "")
original_dataset = "dataset/dalle_pilot/"
target_dataset = "dataset/dalle_4/"
task_name = "clean_dishes"
task_name = "serve_breakfast"
# actions = ['find_fridge', 'open_fridge', 'find_apple', 'pickup_apple', 'find_knife', 'pickup_knife', 'cutintohalf_apple']
actions = ['find_plate', 'pickup_plate', 'find_faucet', 'wash_plate']
actions = ['find_bread', 'pickup_bread', 'find_plate', 'placeon_bread_plate', 'find_tv', 'turnon_tv']
labels = ['0', '2']
dalle_num = 4

def augment():
    # os.mkdir(target_dataset)
    os.mkdir(target_dataset+task_name)
    for a in actions:
        os.mkdir(target_dataset+task_name+"/"+a)
        for label in labels:
            os.mkdir(target_dataset+task_name+"/"+a+"/"+label+'/')
            for ori_im in os.listdir(original_dataset+task_name+"/"+a+"/"+label+'/'):
                ori_im_f = original_dataset+task_name+"/"+a+"/"+label+'/'+ str(ori_im)
                target_im_f = target_dataset+task_name+"/"+a+"/"+label+'/'+ str(ori_im)
                target_im_f = target_im_f.split('.')[0]+".png"
                img = cv2.imread(ori_im_f)
                img = cv2.resize(img, (1024, 1024))
                cv2.imwrite(target_im_f, img, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                response = openai.Image.create_variation(
                  image=open(target_im_f, "rb"),
                  n=dalle_num,
                  size="1024x1024"
                )
                
                for num in range(dalle_num):
                    image_url = response['data'][num]['url']
                    urllib.request.urlretrieve(image_url, target_im_f.split('.')[0]+'dalle'+str(num)+'.png' )

augment()





