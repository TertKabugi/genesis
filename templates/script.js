// Import the functions you need from the SDKs you need
  import { initializeApp } from "https://www.gstatic.com/firebasejs/9.15.0/firebase-app.js";
  import { getAnalytics } from "https://www.gstatic.com/firebasejs/9.15.0/firebase-analytics.js";
  // TODO: Add SDKs for Firebase products that you want to use
  // https://firebase.google.com/docs/web/setup#available-libraries

  // Your web app's Firebase configuration
  // For Firebase JS SDK v7.20.0 and later, measurementId is optional
  const firebaseConfig = {
    apiKey: "AIzaSyD-2iUpc4E1XpseUd2V1XIHW7UEqS1I7hA",
    authDomain: "genesis-ml-369dd.firebaseapp.com",
    projectId: "genesis-ml-369dd",
    storageBucket: "genesis-ml-369dd.appspot.com",
    messagingSenderId: "1049094202128",
    appId: "1:1049094202128:web:e9d0e1bcbf1ff3cab77b95",
    measurementId: "G-NS21KCG176"
  };

  // Initialize Firebase
  const app = initializeApp(firebaseConfig);
  const analytics = getAnalytics(app);
  const auth = firebase.auth()
  const database = firebase.database()

  //Register
  function register (){
    email = document.getElementById('email').value
    username = document.getElementById('name').value
    password = document.getElementById('password').value
  }

  //Create User
  auth.createUserWithEmailAndPassword(email,password)
  .then(function() {
        //Declare User Variable
        var user = auth.currentUser

        //Add user to database
        var database_ref = database.ref()

        //create user data
        var user_data = {
            email : email,
            username : username,
            password: password,
            last_login : Date.now()

        }

        database_ref.child('users/' + user.uid).set(user_data)
        alert('User Created')
  })
  .catch(function(error) {
        var error_code  = error.code
        var error_message = error.message

        alert(error_message)
  })