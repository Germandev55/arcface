import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank


###################################################################################
from flask import Flask, request, jsonify
import datetime
import time
import base64
import numpy as np
import pickle

##############################################################
from threading import Thread

class arcface_NN():
    def __init__(self, mtcnn_ins, learner_ins):
        # starting NN initialize captures with None    
        self.cap_01 = None          # cv captures 
        self.cap_02 = None
        self.cap_01_grab = None     # cv grabs
        self.cap_02_grab = None
        self.cap_01_buffer = None
        self.cap_02_buffer = None
        self.mtcnn = mtcnn_ins
        self.learner = learner_ins
        self.grabs_loop_activated = False   # flag var for loop

    def set_capturess(self, cap_01_address, cap_02_address):
        # TO DO # 
        self.cap_01 = cv2.VideoCapture(cap_01_address)
        self.cap_02 = cv2.VideoCapture(cap_02_address)
        print("captures", self.cap_01.isOpened(), self.cap_02.isOpened())
        # # # # #

    def grabs_loop(self):
        while self.grabs_loop_activated:
            self.cap_01_grab = self.cap_01.grab()
            self.cap_02_grab = self.cap_02.grab()        

    def grabs_loop_start(self):
        self.grab_loop_activated = True
        self.grab_thread = Thread(target=self.grabs_loop, name="grabs_loop_thread")
        self.grab_thread.start()
        print("grab thread started")

    def grabs_loop_stop(self):
        self.grab_loop_activated = False
        self.grab_thread.join()
        print("grab thread joined")
        self.cap_01.release()
        self.cap_02.release()
        print("captures relised")

    def get_current_frame_nn_results(self):
        
        try:
            ret_01, self.cap_01_buffer = self.cap_01.retrieve(self.cap_01_grab)            
            ret_02, self.cap_02_buffer = self.cap_02.retrieve(self.cap_02_grab)            
            
            print("retrives done ", ret_01, ret_02)
            print(self.cap_01_buffer.shape)
            print(self.cap_02_buffer.shape)
            cv2.imshow("cap_01", self.cap_01_buffer)
            cv2.imshow("cap_02", self.cap_02_buffer)
            
            cam_01_ans = self.image_to_nn_data(self.cap_01_buffer)
            cam_02_ans = self.image_to_nn_data(self.cap_02_buffer)

            return {"chanel_01":cam_01_ans, "chanel_02":cam_02_ans}
        except:
            return {"status":"something goes wrong"}

    def image_to_nn_data(self, cv2_image):

        try:
            print("into image to NN")
            
            img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            print("PIL done")
            bboxes, faces = self.mtcnn.align_multi(im_pil, conf.face_limit, conf.min_face_size)
            #print(bboxes, faces)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice    
            print(bboxes)
            # start_time = time.time()
            start_time = time.time()
            results = self.learner.infer_embs(conf, faces, args.tta)
            results_np = results.cpu().detach().numpy()
            print("infer_time", time.time()-start_time)
                        
            return {"status":"done", "coords":bboxes, "embs":results_np}

        except:
            print('detect error')    
            return {"status":"detect error"}

##############################################################
# 
# 

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "these aren't the droids you're looking for"

@app.route('/start', methods=['POST'])
def start():  
    if request.method == 'POST':
        arcface_ins.set_capturess(0, "http://77.243.103.105:8081/mjpg/video.mjpg")
        time.sleep(5)
        arcface_ins.grabs_loop_start()
        return jsonify({"message":"arcface initialized"})

@app.route('/get', methods=['GET'])
def get_frame_data():  
    if request.method == 'GET':
        ret = arcface_ins.get_current_frame_nn_results()
        return jsonify(ret)



if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.00, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-p", "--port", help="port", default=5000, type=int)
    args = parser.parse_args()

    conf = get_config(False)
    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
        print("work on CPU")
    else:
        learner.load_state(conf, 'final.pth', True, True)
        print("work on GPU")
    learner.model.eval()
    print('learner loaded')
    
    arcface_ins = arcface_NN(mtcnn, learner)
    
    # if args.update:
    #     targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
    #     print('facebank updated')
    # else:
    #     targets, names = load_facebank(conf)
    #     print('facebank loaded')

    # cap = cv2.VideoCapture(0)
    # cap.set(3,640)
    # cap.set(4,480)
    
    app.run(host="0.0.0.0", port=args.port) 

