import firebase_admin
from firebase_admin import credentials, db



firebaseConfig = {
    'apiKey': "AIzaSyCGmeVM6OPnRbVGaW6O7DVqafArIGEm5Ys",
    'authDomain': "silwalk-inc.firebaseapp.com",
    'projectId': "silwalk-inc",
    'databaseURL': "https://silwalk-inc-default-rtdb.firebaseio.com",
    'storageBucket': "silwalk-inc.appspot.com",
    'messagingSenderId': "665210785578",
    'serviceAccount': "ServiceKey.json",
    'appId': "1:665210785578:web:6279247f0704ec73be5853",
    'measurementId': "G-EW5W77X7X7"
}

cred = credentials.Certificate("ServiceKey.json")
firebase_admin.initialize_app(cred, firebaseConfig)

rtdb = db.reference()