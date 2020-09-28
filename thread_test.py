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

class grab_thread(Thread):
    def __init__ (self, cv_cap):
        super().__init__()
        self.grab_loop_activated = False
        self.cap = cv_cap
        print("thread initialized")

    def run(self):
        self.grab_loop_activated = True
        print("grab loop started", self.name)
        while self.grab_loop_activated:
            self.cap_grab = self.cap.grab()

    def get_grab(self):
        return self.cap_grab

    def stop_loop(self):
        self.grab_loop_activated = False

class arcface_NN():
    def __init__(self, mtcnn_ins, learner_ins):
        # starting NN initialize captures with None    
        self.cap_01 = None          # cv captures 
        self.cap_02 = None
        self.cap_01_buffer = None
        self.cap_02_buffer = None
        self.mtcnn = mtcnn_ins
        self.learner = learner_ins

    def set_capturess(self, cap_01_address, cap_02_address):
        # TO DO # 
        self.cap_01 = cv2.VideoCapture(cap_01_address)
        self.cap_02 = cv2.VideoCapture(cap_02_address)
        print("captures", self.cap_01.isOpened(), self.cap_02.isOpened())


    # def grabs_loop(self):
    #     while self.grabs_loop_activated:
    #         self.cap_01_grab = self.cap_01.grab()
    #         self.cap_02_grab = self.cap_02.grab()        
    #         print ("grabs loop works")
    #         # print(self.cap_01_grab, self.cap_02_grab)

    def grabs_loop_start(self):
        self.grab_thread1 = grab_thread(self.cap_01)
        self.grab_thread2 = grab_thread(self.cap_02)
        self.grab_thread1.start()
        self.grab_thread2.start()
        print("grab threads started")

    def grabs_loop_stop(self):
        self.grab_thread1.stop_loop()
        self.grab_thread2.stop_loop()
        self.grab_thread1.join()
        self.grab_thread2.join()

        print("grab thread joined")
        self.cap_01.release()
        self.cap_02.release()
        print("captures relised")

    def get_current_frame_nn_results(self):
        
        try:
            ret_01, self.cap_01_buffer = self.cap_01.retrieve(self.grab_thread1.get_grab())            
            ret_02, self.cap_02_buffer = self.cap_02.retrieve(self.grab_thread2.get_grab())            
            
            print("retrives done ", ret_01, ret_02)
            print(self.cap_01_buffer.shape)
            print(self.cap_02_buffer.shape)
            #cv2.imshow("cap_01", self.cap_01_buffer)
            #cv2.imshow("cap_02", self.cap_02_buffer)
            
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

            bboxes_py = bboxes.tolist()
            results_py = results_np.tolist()
            
            return {"status":"done", "coords":bboxes_py, "embs":results_py}

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
        data = request.get_json(force=True)
        chanel_01_address = data["chanel_01"]
        chanel_02_address = data["chanel_02"]
        arcface_ins.set_capturess(chanel_01_address, chanel_02_address)
        time.sleep(5)
        arcface_ins.grabs_loop_start()
        return jsonify({"message":"arcface initialized"})

@app.route('/get', methods=['GET'])
def get_frame_data():  
    if request.method == 'GET':
        ret = arcface_ins.get_current_frame_nn_results()
        print(ret)
        return jsonify(ret)

if __name__ == '__main__':
    
    ### TEST #####
    # cap01 = cv2.VideoCapture(0)
    # cap02 = cv2.VideoCapture("rtsp://admin:Admin1234@92.125.152.58:6461")
    # cap_thread1 = grab_thread(cap01)
    # cap_thread2 = grab_thread(cap02)
    
    # cap_thread1.start()
    # cap_thread2.start()
    # #cap_thread2 = grab_thread(cv2.VideoCapture("http://77.243.103.105:8081/mjpg/video.mjpg"))
    
    # time.sleep(10)
    
    # _, frame1 = cap01.retrieve(cap_thread1.cap_grab)
    # _, frame2 = cap02.retrieve(cap_thread2.cap_grab)
    # cv2.imshow("frame1", frame1)
    # cv2.imshow("frame2", frame2)


    # cap_thread1.grab_loop_activated = False
    # cap_thread2.grab_loop_activated = False

    # cap_thread1.join()
    # cap_thread2.join()
    # time.sleep(2)
    
    # cap01.release()
    # cap02.release()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # quit()
    ### TEST #####
    
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
    
    ###########################################################################################

    # arcface_ins.set_capturess(0, "rtsp://admin:Admin1234@92.125.152.58:6461")
    # time.sleep(5)
    # arcface_ins.grabs_loop_start()

    # time.sleep(5)
    # print(arcface_ins.get_current_frame_nn_results())
    # time.sleep(5)
    # arcface_ins.grabs_loop_stop()
    # print("stopped")
    
    # # #######################################################################################

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
