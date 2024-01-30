import cv2
import os
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import numpy as np
from deepface import DeepFace
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2

app = Flask(__name__) 
# for dealing with cache storing issue of browser
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = 'static/uploaded_files'


face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
classifier = load_model('models/model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# to generate webcam stream
def generate_frames():
    camera=cv2.VideoCapture(0)
    while True: 
        success,frame=camera.read()
        if not success:
            break 
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        # yield will continously generate webcam stream (frame by frame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# making route for homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")

# making route for homepage
@app.route('/use_app', methods=['GET', 'POST'])
def use_app():
    return render_template("index.html")

#handling images
@app.route('/handle_image', methods=['GET', 'POST'])
def handle_image():
    # getting image 
    f = request.files['file1']
    # renaming image
    f.filename = "input_image.jpg"
    # saving image in static folder 
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

    frame = cv2.imread("static/uploaded_files/input_image.jpg")
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)
            dominant_emotion = result[0]["dominant_emotion"]
            label=emotion_labels[prediction.argmax()]
            label = label + " " + str(round(float(result[0]["emotion"][dominant_emotion]), 2))
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imwrite('static/saved_predictions/output_image.jpg', frame)
        
    return render_template("image.html")


#handle webcam live streaming prediction
# to generate webcam stream
def generate_frames_for_webcam_live_stream_prediction():
    camera=cv2.VideoCapture(0)
    while True: 
        success,frame=camera.read()
        if not success:
            break 
        else:
            labels = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    prediction = classifier.predict(roi)[0]
                    result = DeepFace.analyze(frame, enforce_detection=False, actions = ['emotion'])
                    dominant_emotion = result[0]["dominant_emotion"]
                    label=emotion_labels[prediction.argmax()]
                    label = label + " " + str(round(float(result[0]["emotion"][dominant_emotion]), 2))
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                else:
                    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        # yield will continously generate webcam stream (frame by frame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_live_streaming_prediction')
def webcam_live_streaming_prediction():
    return Response(generate_frames_for_webcam_live_stream_prediction(),mimetype='multipart/x-mixed-replace; boundary=frame')

#handling webcam
@app.route('/handle_webcam', methods=['GET', 'POST'])
def handle_webcam():
    return render_template("webcam.html")

@app.route('/streaming')
def streaming():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_show_pic', methods=['GET', 'POST'])
def capture_and_show_pic():
    camera=cv2.VideoCapture(0)
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            cv2.imwrite("static/uploaded_files/webcam_image.jpg", frame)
            break
    return "Success"

@app.route('/webcam_image_predict', methods=['GET', 'POST'])
def webcam_image_predict():
    frame = cv2.imread("static/uploaded_files/webcam_image.jpg")
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            result = DeepFace.analyze(frame, enforce_detection=False, actions = ['emotion'])
            dominant_emotion = result[0]["dominant_emotion"]
            label=emotion_labels[prediction.argmax()]
            label = label + " " + str(round(float(result[0]["emotion"][dominant_emotion]), 2))
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imwrite('static/saved_predictions/output_image.jpg', frame)
    return render_template("image.html")




# for dealing with cache storing issue of browser
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True) 