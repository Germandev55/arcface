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

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():  
    all_time = time.time()
    if request.method == 'POST':
        # проверяем, что прислали файл
        # if 'file' not in request.files:
        #    return "someting went wrong 1"
  
        # data = request.get_json(force=True)
        data = request.get_data()
        # user_file = request.files['file']
        # temp = request.files['file']
        # if user_file.filename == '':
        #     return "file name not found ..." 

        try:

            # print(data_frames)
            #frame_bytes = data_frames["cam_1"][2:-1].encode('utf-8')
            #print(frame_bytes)
            print(type(data))
            unpick = pickle.loads(data)
            print(unpick.shape)
            # print(data)
            # frame_bytes = frame_bytes["cam_1"]
            # print(frame_bytes)
            #print(jpg_data, "jpg_data done")
            image = Image.fromarray(unpick)
            start_time = time.time()
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            print("mtcnn time", time.time()-start_time)
            #print(bboxes, faces)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice    
            print(bboxes)
            # start_time = time.time()
            start_time = time.time()
            results = learner.infer_embs(conf, faces, args.tta)
            results_np = results.cpu().detach().numpy()
            print("infer_time", time.time()-start_time)
            # results_np = results.numpy()
            # print(results_np)
            # print("face_infer time", time.time()-start_time)
            # for idx,bbox in enumerate(bboxes):
            #     if args.score:
            #         frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
            #     else:
            #         frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            # cv2.imshow('face Capture'+str(args.port), unpick)
            # cv2.waitKey(1)
            ret = {"coords":bboxes, "embs":results_np}
            ret_bytes = pickle.dumps(ret)
            print("post time ", time.time()-all_time)    
            return ret_bytes

        except:
            print('detect error')    
            return jsonify({"status":"detect error"})

        # return jsonify({
        #     "status":"success",
        #     "prediction":"XXX",
        #     "upload_time":"11223344"
        #     })


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
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