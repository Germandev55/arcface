from PyQt5.QtCore import  QThread
import cv2
from flask import Flask,Response,request
from flask import send_file
import requests
import json
import numpy as np
import Global_vars

class Start_server(QThread):    
    def __init__(self, parent=None):
        super(Start_server, self).__init__(parent)
    def run(self):
        app = Flask(__name__) #static_folder='static'
        app.config['SECRET_KEY'] = 'any secret string'
        app.run(host='0.0.0.0', debug=False, use_reloader=False)
        print('flask work')

class Control_thread(QThread):
    print('into Control_thread')
    def __init__(self, parent=None):
        super(Control_thread, self).__init__(parent)
    def run(self):
        while True:
            try:           
                r = requests.get("http://localhost:5000/get")
                print("status ___ ", r.status_code)
                print('get__ ', r.url)
            except:
                print("requests.get failed")

     
                
        
        