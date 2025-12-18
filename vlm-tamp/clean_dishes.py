import os
import time
from random import sample, uniform
# from allennlp.predictors.predictor import Predictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import requests
# from huggingface_hub import snapshot_download
from transformers import BlipProcessor, BlipForQuestionAnswering, Blip2Processor, Blip2ForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
device = 'cuda'
# processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-capfilt-large")
# model = BlipForQuestionAnswering.from_pretrained("ybelkada/blip-vqa-capfilt-large")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map="auto")

root_path = "datasets/tpvqa_dalle_v0/dalle_4/clean_dishes/"
task_path = "datasets/tpvqa_dalle_v0/dalle_4/clean_dishes/"
actions = ['find_plate', 'pickup_plate', 'find_faucet', 'wash_plate']
image_cls = ['0', '2'] 
unknown = ['xxx', 'yyy', 'zzz']
# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/vilbert-vqa-pretrained.2021-03-15.tar.gz")
objects = ['plate', 'faucet']
preconditions = {
        'find': [],   
        'pickup': ['Is there xxx?'],
        'wash': ['Is xxx in robot\'s hand?'],
        'placeon': ['Is xxx in robot\'s hand?'], 
        'turnon': ['Is there xxx?'], 
        'open': ['Is there xxx?'], #TODO: is closed?
        'cutintohalf': ['Is there xxx?'], 
        }
effects = {
        'find': ['Is there xxx?'],
        'pickup': ['Is xxx in robot\'s hand?'],
        'wash': ['Is xxx clean?'],
        'placeon': ['Is xxx on yyy?'],
        'turnon': ['Is xxx on?'],
        'open': ['Is xxx open?'],
        'cutintohalf': ['Is xxx cut into half?'],
        }

success_effects = {
        'find': ['Was find xxx successful?'],
        'pickup': ['Was pick up xxx successful?'],
        'wash': ['Was wash xxx successful?'],
        'placeon': ['Was place xxx on yyy successful?'],
        'turnon': ['Was turn on xxx successful?'],
        'open': ['Was open xxx successful?'],
        'cutintohalf': ['Was cut xxx into half successful?'],
        }

success_preconditions = {
        'find': ['Is is possible to find xxx here?'],
        'pickup': ['Is is possible to pick up xxx here?'],
        'wash': ['Is is possible to wash xxx here?'],
        'placeon': ['Is it possible to place xxx on yyy here?'],
        'turnon': ['Is it possible to turn on xxx here?'],
        'open': ['Is it possible to open xxx here?'],
        'cutintohalf': ['Is it possible to cut xxx into half here?'],
        }
        
prev_ac = {
        'pickup_plate': 'find_plate',
        'wash_plate': 'pickup_plate',
        }

def sample_obs(a_name):
    all_images = []
    for i_cls in image_cls:
        if i_cls != 'self':
            for im in os.listdir(task_path+a_name+'/'+i_cls+'/'):
                all_images.append((im,i_cls))
                if i_cls == '0':   # increase the chance for sample the correct case
                    all_images.append((im,i_cls))
                    all_images.append((im,i_cls))
    ret = sample(all_images, 1)[0]
    return ret

def sample_obs_failed(a_name):
    all_images = []
    for i_cls in image_cls:
        if i_cls == '2':
            for im in os.listdir(task_path+a_name+'/'+i_cls+'/'):
                all_images.append((im,i_cls))
    return sample(all_images, 1)[0]

def check_preconditions(a_name, file_path):
    objects = []
    for tok in a_name.split('_')[1:]:
        if not tok.isnumeric(): # not an action with the same name
            objects.append(tok)
    is_satisfied = True
    for question_temp in preconditions[a_name.split('_')[0]]:
        question = question_temp
        for un_tok_index in range(len(unknown)):
            un_tok = unknown[un_tok_index]
            if un_tok in question_temp:
                question = question.replace(un_tok, objects[un_tok_index])
        # print ("Ask question for Precondition: "+question+" on image: "+file_path)
        # result = predictor.predict(
        #     question=question,
        #     image=file_path
        # )
        # if result['tokens']['yes']<result['tokens']['no']:
        #     is_satisfied = False
            # print ("Answer: No")
        # else:
            # print ("Answer: Yes")
        raw_image = Image.open(file_path).convert('RGB')

        inputs = processor(raw_image, question, return_tensors="pt")

        out = model.generate(**inputs)
        if processor.decode(out[0], skip_special_tokens=True) == "no":
            is_satisfied = False
    return is_satisfied



def check_effects(a_name, file_path):
    objects = []
    for tok in a_name.split('_')[1:]:
        if not tok.isnumeric(): # not an action with the same name
            objects.append(tok)
    is_satisfied = True
    for question_temp in effects[a_name.split('_')[0]]:
        question = question_temp
        for un_tok_index in range(len(unknown)):
            un_tok = unknown[un_tok_index]
            if un_tok in question_temp:
                question = question.replace(un_tok, objects[un_tok_index])
        # print ("Ask question for Effect: "+question+" on image: "+file_path)
        # result = predictor.predict(
        #     question=question,
        #     image=file_path
        # )
        # if result['tokens']['yes']<result['tokens']['no']:
        #     is_satisfied = False    main()
            # print ("Observed answer: No")
        # else:
            # print ("Observed answer: Yes")
        raw_image = Image.open(file_path).convert('RGB')

        inputs = processor(raw_image, question, return_tensors="pt")

        out = model.generate(**inputs)
        if processor.decode(out[0], skip_special_tokens=True) == "no":
            is_satisfied = False
    return is_satisfied

def eval_vqa_action_effect(a, timeout):
    start_time = time.time()
    total = 0
    fail = 0
    while True:
        if time.time()-start_time>timeout:
            break
        im, gt_index = sample_obs(a)
        pred = check_effects(a, root_path+a+'/'+gt_index+'/'+im)
        gt_success = True
        if gt_index == '2':
            gt_success = False
        if pred != gt_success:
            fail += 1
        total += 1
    return 1-float(fail)/float(total)
        
def eval_vqa_action_precondition(a, timeout):
    start_time = time.time()
    total = 0
    fail = 0
    while True:
        if time.time()-start_time>timeout:
            break
        if actions.index(a) == 0:
            return 1
        if preconditions[a.split('_')[0]] == []:
            return 1
        im, gt_index = sample_obs(prev_ac[a])
        pred = check_preconditions(a, root_path+prev_ac[a]+'/'+gt_index+'/'+im)
        gt_success = True
        if gt_index == '2':
            gt_success = False
        if pred != gt_success:
            fail += 1
        total += 1
    return 1-float(fail)/float(total)
 

def check_success_effects(a_name, file_path):
    objects = []
    for tok in a_name.split('_')[1:]:
        if not tok.isnumeric(): # not an action with the same name
            objects.append(tok)
    is_satisfied = True
    for question_temp in success_effects[a_name.split('_')[0]]:
        question = question_temp
        for un_tok_index in range(len(unknown)):
            un_tok = unknown[un_tok_index]
            if un_tok in question_temp:
                question = question.replace(un_tok, objects[un_tok_index])
        # print ("Ask question for Effect: "+question+" on image: "+file_path)
        # result = predictor.predict(
        #     question=question,
        #     image=file_path
        # )
        # if result['tokens']['yes']<result['tokens']['no']:
        #     is_satisfied = False
            # print ("Observed answer: No")
        # else:
            # print ("Observed answer: Yes")
        raw_image = Image.open(file_path).convert('RGB')

        inputs = processor(raw_image, question, return_tensors="pt")

        out = model.generate(**inputs)
        if processor.decode(out[0], skip_special_tokens=True) == "no":
            is_satisfied = False
    return is_satisfied
    
def check_success_preconditions(a_name, file_path):
    objects = []
    for tok in a_name.split('_')[1:]:
        if not tok.isnumeric(): # not an action with the same name
            objects.append(tok)
    is_satisfied = True
    for question_temp in success_preconditions[a_name.split('_')[0]]:
        question = question_temp
        for un_tok_index in range(len(unknown)):
            un_tok = unknown[un_tok_index]
            if un_tok in question_temp:
                question = question.replace(un_tok, objects[un_tok_index])
        # print ("Ask question for Effect: "+question+" on image: "+file_path)
        # result = predictor.predict(
        #     question=question,
        #     image=file_path
        # )
        # if result['tokens']['yes']<result['tokens']['no']:
        #     is_satisfied = False
            # print ("Observed answer: No")
        # else:
            # print ("Observed answer: Yes")
        raw_image = Image.open(file_path).convert('RGB')

        inputs = processor(raw_image, question, return_tensors="pt")

        out = model.generate(**inputs)
        if processor.decode(out[0], skip_special_tokens=True) == "no":
            is_satisfied = False
    return is_satisfied   
'''
Evaluation starts here =====================================>
'''
def main():
    demo = False
    
    baseline = False
    successVQA = False
    palme = False
    effect_only = True
    
    uncertainty = True # a failed action effects previous observation
    
    prob = 0.3 # the probability of uncertainty.
    max_run = 100
    
    run_idx = 0
    retrying_count = 0
    replanning_count = 0
    success_run = 0
    failed_run = 0
    
    if successVQA or baseline or effect_only:
        check_pre = False
    else:
        check_pre = True
    
    while run_idx < max_run:
        print("============ Trial "+str(run_idx)+" ============")
        success_action_count = 0
        prev_action = None
        prev_obs = None
        replanning = False
        for a in actions:
            print (a)
            
            # baseline performance
            if baseline:
                gt_success = False
                cur_obs, gt_index = sample_obs(a)
                if gt_index == '0' or gt_index == '1':
                    gt_success = True
            
            # ours
            else:
                has_retried = False
                while True:
                    # After taking an action, sample an observation
                    cur_obs, gt_index = sample_obs(a)
                    if demo:
                        img = mpimg.imread(root_path+"all_images/"+cur_obs)
                        imgplt = plt.imshow(img)
                        plt.show()
                    
                    if uncertainty and has_retried:
                        # under some probability to sample the previous state
                        if a in prev_ac and uniform(0,1)<prob:

                            prev_obs, prev_gt_index = sample_obs_failed(prev_ac[a])
                        
                            if not check_pre:
                                if prev_gt_index == '2':
                                    print ("Some precondition not satisfied. Stop retrying.")
                                    gt_success = False
                                    break
                            else:
                                if palme:
                                    pred_success = check_success_preconditions(a, root_path+prev_ac[a]+'/'+prev_gt_index+'/'+prev_obs)                                
                                else:
                                    pred_success = check_preconditions(a, root_path+prev_ac[a]+'/'+prev_gt_index+'/'+prev_obs)
                                if not pred_success:
                                    replanning = True
                                    print ("Replanning triggerd! According to the new plan, redo from the beginning")
                                    replanning_count += 1
                                    break

                    gt_success = True
                    if gt_index == '2':
                        gt_success = False
                    
                    # Our method: Check action effects
                    # print (cur_obs)
                    if successVQA or palme:
                        pred_success = check_success_effects(a, root_path+a+'/'+gt_index+'/'+cur_obs)
                    else:
                        pred_success = check_effects(a, root_path+a+'/'+gt_index+'/'+cur_obs)

                    if pred_success:
                        break
                    else:
                        print ("Action failed. Retrying...")
                        has_retried = True
                        retrying_count += 1
            if replanning:
                break
            if not gt_success:
                print ("Task failed :(")
                failed_run += 1
                run_idx += 1
                break
            else:
                success_action_count += 1
                prev_obs = cur_obs
                prev_gt_index = gt_index
                prev_action = a
        
        if success_action_count == len(actions):
            success_run += 1
            run_idx += 1

    print ("Success: " + str(success_run))
    print ("Failure: " + str(failed_run))
    print ("Retrying due to effects check: "+str(retrying_count))
    print ("Replanning due to preconditions check: "+str(replanning_count))
if __name__ == "__main__":
    for a in actions:
        print (a, eval_vqa_action_effect(a, 10))

    main()




