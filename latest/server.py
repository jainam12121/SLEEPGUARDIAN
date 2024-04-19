from datetime import timedelta
import redis
from flask_session import Session
from flask import Flask, redirect, request, render_template, jsonify,session
import os
import cv2
import numpy as np
# from playsound import playsound
import base64
from werkzeug.middleware.proxy_fix import ProxyFix
import dlib
from pymongo import MongoClient
from bson.objectid import ObjectId
from scipy.spatial import distance as dist
from dateutil.parser import parse
from datetime import datetime
from flask import get_flashed_messages
from flask import flash
import re
from flask_login import LoginManager
from twilio.rest import Client

try:
    cluster = MongoClient(  "mongosh "mongodb+srv://dds.rpi9euq.mongodb.net/" --apiVersion 1 --username jainamdp2002")
    db = cluster["Cluster0"]
    orgg = db["organisation"]
    
    print("Database connected successfully")
except Exception as e:
    print("An error occurred while connecting to the database: ", e)
# db.organisation.create_index("username", unique=True)
# db.organisation.create_index("email", unique=True)
# db.individual.create_index("username", unique=True)
# db.individual.create_index("email", unique=True)
# db.individual.drop_index("username_1")
db.individual.create_index("username")
# db.individual.drop_index("email_1")
db.individual.create_index("email")
# db.organisation.drop_index("username_1")
db.organisation.create_index("username")
# db.organisation.drop_index("email_1")
db.organisation.create_index("email")

app = Flask(__name__)
 # Or another session type like 'redis'
# app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')
# session(app)
# app.config["SESSION_PERMANENT"] = True
# app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30) # Set session lifetime



# app.config['SECRET_KEY'] = '3f28fccfea0476c6b7c6c2ffa02c0e4b' # Make sure to set a secret key

# sess = Session()
# sess.init_app(app)


# app.config ['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')

# Initialize Flask-Session within an application context
# with app.app_context():
#     sess = Session()
#     sess.init_app(app)

app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.wsgi_app = ProxyFix(app.wsgi_app)
app.secret_key = '3f28fccfea0476c6b7c6c2ffa02c0e4b'
secret_key = os.urandom(24)
# Load the pre-trained models and define constants
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
EYE_AR_THRESH = 0.22
MOUTH_OPEN_THRESH = 0.55
account_sid = 'AC1ab6a681eaac1ef37e7d40fc88399d4d'
auth_token = 'd92c830a4697dccb7c620c3f91c4b5fe'
client = Client(account_sid, auth_token)
# Define the calculate_ear and calculate_mar functions
def calculate_ear(eye):
 p2_minus_p6 = dist.euclidean(eye[1], eye[5])
 p3_minus_p5 = dist.euclidean(eye[2], eye[4])
 p1_minus_p4 = dist.euclidean(eye[0], eye[3])
 ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
 return ear

def calculate_mar(mouth):
 if len(mouth) < 10:
    return 1e308
 p6_minus_p2 = dist.euclidean(mouth[5], mouth[1])
 p10_minus_p4 = dist.euclidean(mouth[9], mouth[3])
 p1_minus_p7 = dist.euclidean(mouth[0], mouth[6])
 mar = (p6_minus_p2 + p10_minus_p4) / (2.0 * p1_minus_p7)
 return mar



# Initialize the detector and counter
detector = dlib.get_frontal_face_detector()
COUNTER = 0

# Add a new route to process incoming frames for drowsiness detection
@app.route('/process_frame', methods=['POST'])
def process_frame():
 try:
  # Decode the image from base64 format
  data = request.get_json()
  encoded_data = data['image'].split(',')[1]
  nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  print(f'Number of faces detected: {len(faces)}') # Print the number of faces
  
  response_data = {
           'num_faces': len(faces),
           'ears': [],
           'mars': [],
           'drowsiness_detected': False
       }
  
  # Detect faces in the image
  if len(faces) > 0:
   for (x, y, w, h) in faces:
    # Extract the region of interest (ROI) containing the face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    left_ear = 0
    right_ear = 0
    mar = 0
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # Perform drowsiness detection on the ROI
    rects = detector(roi_gray, 1)
    for rect in rects:
      shape = predictor(roi_gray, rect)
      shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
      print(f'Shape length: {len(shape)}') 
      # Get the landmarks for the left and right eyes
      if len(shape) >= 68:
       leftEye = shape[36:42]
       rightEye = shape[42:48]
       mouth = shape[60:68]
      if len(shape) >= 68:
       mouth = shape[48:67]
       mar = calculate_mar(mouth)
       # Calculate the EAR for each eye
       left_ear = calculate_ear(leftEye)
       right_ear = calculate_ear(rightEye)
      #  mar = calculate_mar(mouth)
       ear = (left_ear + right_ear) / 2.0
       print(f'ear: {ear}')
       print(f'mar {mar}')
       # Append the EAR and MAR to response_data
       response_data['ears'].append((left_ear + right_ear) / 2.0)
       response_data['mars'].append(mar)
       # Assuming you have the individual's ID or some unique identifier
       user_id = session.get('user_id')
       ind_cont = db.individual.find_one({'_id': ObjectId(user_id)})['ind_cont']

      # Check to see if the eye aspect ratio is below the blink threshold
      if ((left_ear + right_ear) / 2.0 < EYE_AR_THRESH ) or mar > MOUTH_OPEN_THRESH: 
        response_data['drowsiness_detected'] = True

        # playsound('/path/to/your/sound/file.mp3', block=False)
        message = client.messages.create(
            body="Drowsiness detected. Please take a break.",
            from_="+12315447420", # Your Twilio phone number
            to=ind_cont # Driver's phone number
        )
        print(f"Message sent to driver: {message.sid}")

            # # Send SMS to the organization
            # message = client.messages.create(
            #     body="Drowsiness detected for a driver. Please check.",
            #     from_="+12315447420", # Your Twilio phone number
            #     to=org_cont # Organization's phone number
            # )
            # print(f"Message sent to organization: {message.sid}")
      else:
       COUNTER = 0
       response_data['drowsiness_detected'] = False

  return jsonify(response_data)
 except Exception as e:
  return jsonify({'error': str(e)})

@app.route('/set_session')
def set_session():
    session['key'] = 'value'
    return 'Session data set'

@app.route('/register')
def index():
 return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('_flashes', None)
    return redirect('index.html')

@app.route('/camera')
def camera():
 
 flash('Login successful')
 user_id = session.get('user_id')
 user_type = session.get('user_type')
 data = None
 print(f'User ID: {user_id}, User Type: {user_type}')

 if user_type == '1':
      collection = db['individual']
 elif user_type == '2':
      collection = db['organisation']
 else:
      return "Invalid user type", 400

  # Find the user in the database
 user = collection.find_one({'_id': ObjectId(user_id)})

  # Check if user exists
 if user is None:
      return "User does not exist", 404

 data = user
 print(data)

 return render_template('camera.html',data=data, user_type=user_type)


@app.route('/afterlogin', methods=['GET'])
def afterlogin():
  # Retrieve the user's ID from the session
  user_id = session.get('user_id')
  user_type = session.get('user_type')
  data = None
  print(f'User ID: {user_id}, User Type: {user_type}')

  if user_type == '1':
      collection = db['individual']
  elif user_type == '2':
      collection = db['organisation']
  else:
      return "Invalid user type", 400

  # Find the user in the database
  user = collection.find_one({'_id': ObjectId(user_id)})

  # Check if user exists
  if user is None:
      return "User does not exist", 404

  data = user
  print(data)

  return render_template('afterlogin.html', data=data, user_type=user_type)

   





@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        if not re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', email):
           flash('Invalid email address.')
           return redirect(request.url)
        user_type = request.form.get('option')  # '1' for individual, '2' for organization

        # Determine the correct collection and password field based on user type
        if user_type == '1':  # Individual
            collection = db['individual']
            password_field = 'ind_password'
            email_field = 'ind_email'
        elif user_type == '2':  # Organization
            collection = db['organisation']  # Ensure this is the correct collection name
            password_field = 'org_password'
            email_field = 'org_email'
        else:
            return "Invalid user type", 400

        # Find the user in the database
        user = collection.find_one({email_field: email})
        # print(f'User: {user}')
        # print(f'Password Field Value: {user[password_field]}')
        # Check if user exists and password matches
        if user and user[password_field] == request.form.get('password'):
           session['user_id'] = str(user['_id'])
           session['user_type'] = user_type
          
          
           print("hi")
           print(session.keys(), session.values)

           return redirect('/camera')
            # Login successful
            
        else:
            # Login failed
            return "Invalid credentials", 401
    
    return render_template('index.html')



@app.route('/individualregister', methods=['GET', 'POST'])
def individual_register():
 if request.method == 'POST':
  try:
      # Extract form data
      ind_name = request.form.get('ind_name')
      # username = request.form.get('username')
      ind_email = request.form.get('ind_email')
      if not re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ind_email):
          
           flash('Invalid email address.')
           return redirect(request.url)
      ind_gender = request.form.get('ind_gender')
      ind_DOB = request.form.get('ind_DOB')
      try:
          dob = parse(ind_DOB)
          if dob > datetime.now():
              raise ValueError
      except ValueError:
          flash('Invalid date of birth.')
          return redirect(request.url)
      ind_cont = request.form.get('ind_cont')
      print(ind_cont)
      if not re.match(r'^(?!0)\d{10}$', ind_cont):
        flash('Contact should accept only numbers up to  10 digits and not start with  0.', 'ind_cont_error')
        return redirect(request.url)
      else:
        print("value is correct")
      ind_ADDRESS = request.form.get('ind_ADDRESS')
      ind_password = request.form.get('ind_password')
      organisation = request.form.get('organization')
      ind_confirm_password = request.form.get('ind_confirm_password')
      if ind_password != ind_confirm_password:
          flash('Passwords do not match.')
          return redirect(request.url)
      username = request.form.get('username') # Add this line
      existing_user = db.individual.find_one({'username': username})
      if existing_user:
          flash('Username already exists.')
          return redirect(request.url)

      
      # Check if username is not None
      if username is None:
          return "Username is required", 400

      # Insert form data into MongoDB collection
      result = db.individual.insert_one({
          'ind_name': ind_name,
          'username': username,
          'ind_email': ind_email,
          'ind_gender': ind_gender,
          'ind_DOB': ind_DOB,
          'ind_cont': ind_cont,
          'ind_ADDRESS': ind_ADDRESS,
          'ind_password': ind_password,
          'ind_confirm_password': ind_confirm_password,
          
          'organisation': organisation
      })
      print("hi")
      print(f"Document inserted with _id: {result.inserted_id}")

      return redirect('/')
  except Exception as e:
      print(f"An error occurred: {e}")
      return str(e), 500
 org_cursor = db.organisation.find({"org_name": {"$exists": True}})
 org_names = [doc['org_name'] for doc in org_cursor]
 return render_template('individualregister.html', organizations=org_names)


@app.route('/organisationregister', methods=['GET', 'POST'])
def organisation_register():
 if request.method == 'POST':
    try:
        # Extract form data
        org_name = request.form.get('org_name')
        username = request.form.get('username')
        existing_user = db.organisation.find_one({'username': username})
        if existing_user:
          flash('Username already exists.')
          return redirect(request.url)
        org_email = request.form.get('org_email')
        if not re.match(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', org_email):
           flash('Invalid email address.')
           return redirect(request.url)
        org_add = request.form.get('org_add')
        org_cont = request.form.get('org_cont')
        print(org_cont)
        if not re.match(r'^(?!0)\d{10}$', org_cont):
            flash('Contact should accept only numbers up to  10 digits and not start with  0.', 'ind_cont_error')
            return redirect(request.url)
        org_password = request.form.get('org_password')
        org_confirm_password = request.form.get('org_confirm_password')
        if org_password != org_confirm_password:
          flash('Passwords do not match.')
          return redirect(request.url)
        org_identity = request.files['org_identity'].read()

        # Insert form data into MongoDB collection
        result = db.organisation.insert_one({
            'org_name': org_name,
            'username': username,
            'org_email': org_email,
            'org_add': org_add,
            'org_cont': org_cont,
            'org_password': org_password,
            'org_confirm_password': org_confirm_password,
            'org_identity': org_identity
        })

        print(f"Document inserted with _id: {result.inserted_id}")

        return redirect('/')
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred", 500

 return render_template('organizationregister.html')





if __name__ == '__main__':
 app.run(debug=True)
