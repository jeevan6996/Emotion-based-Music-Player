# import the Flask class from the flask module
from flask import Flask, render_template, redirect, url_for, request, Response
from flask_mysqldb import MySQL
import os
import random
from camera import VideoCamera

# create the application object
app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root12345'
app.config['MYSQL_DB'] = 'Emoplayer'

mysql = MySQL(app)

user = ""
emotion = ""

@app.route('/')    
@app.route('/layout')    
def layout():
    cur = mysql.connection.cursor()
    cur.execute("call resetLogin")
    mysql.connection.commit()
    cur.close()
    return render_template('layout.html')  # render a template

@app.route('/register', methods=['GET','POST'])    
def register():
    if request.method == "POST":
        details = request.form
        Name=details['name']
        Email=details['email']
        Password=details['pswd']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO UserMaster(name, emailID, password) VALUES (%s, %s, %s)", (Name ,Email, Password))
        cur.execute("SELECT getUserID(%s)",(Email,))
        userID = cur.fetchone()
        userID = userID[0]
        Happy=details['Happy']
        Sadness=details['Sad']
        Anger=details['Angry']
        Neutral=details['Neutral']
        cur.execute("INSERT INTO PrefMaster(userID,happy,sadness,anger,neutral) VALUES (%s, %s, %s, %s, %s)",(int(userID),Happy,Sadness,Anger,Neutral))
        mysql.connection.commit()
        cur.close()
    return render_template('register.html')  # render a template
    
@app.route('/login', methods=['GET', 'POST'])
def login():  
    flag = 1
    Email = ""
    cur = mysql.connection.cursor()
    cur.execute("call resetLogin")
    if request.method == "POST":
        details = request.form
        Email = details['email']
        Password = details['pswd']
        cur = mysql.connection.cursor()
        cur.execute("call resetLogin")
        cur.execute("SELECT checkPassword(%s,%s)", (Email, Password))
        flag = cur.fetchone()
        flag = flag[0]
        if(flag == 1):
            cur.execute("SELECT getUserID(%s)",(Email,))
            userID = cur.fetchone()
            userID = userID[0]
            cur.execute("SELECT name from UserMaster where userID = %s", (int(userID),))
            user = cur.fetchone()
            user = user[0]
            print("User",user)
            cur.execute("UPDATE LoginMaster set status='Y' where userID = %s", (int(userID),))
            mysql.connection.commit()
            cur.close()
            return redirect(url_for('userpage', loggedUser = user))
            #return render_template('userpage.html',loggedUser = user)   #error=error          
    mysql.connection.commit()
    cur.close()
        #return 'success'
    return render_template('login.html', loginFlag = flag , emailID = Email)  #error=error

@app.route('/userpage', methods=['GET', 'POST'])    
def userpage():
    cur = mysql.connection.cursor()
    cur.execute("SELECT getLoggedUserID()")
    userID = cur.fetchone()
    userID = userID[0]
    print("UsrId",userID)
    cur.execute("SELECT name from UserMaster where userID = %s", (int(userID),))
    user = cur.fetchone()
    user = user[0]
    print("User",user)
    mysql.connection.commit()
    cur.close()
    emotion = ""
    if request.method == "POST":
        os.system('python3 videoPredict.py')    
        file1 = open("Emotion.txt","r")
        emotion=file1.read()
        print(emotion)
        file1.close()
    return render_template('userpage.html',loggedUser = user, emotionUser = emotion)  # render a template
'''
@app.route('/camera')    
def camera():
    return render_template('camera.html')  # render a template    	   
'''
 
@app.route('/MusicPlayer')
def MusicPlayer():
    file1 = open("Emotion.txt","r")
    emotion=file1.read()
    print(emotion)
    file1.close()
    #Email="demo@gmail.com"
    cur = mysql.connection.cursor()
    cur.execute("SELECT getLoggedUserID()")
    userID = cur.fetchone()
    userID = userID[0]
    print("UsrId",userID)
    cur.execute("SELECT name from UserMaster where userID = %s", (int(userID),))
    user = cur.fetchone()
    user = user[0]
    print("User",user)
    query="SELECT "+emotion+" from PrefMaster where userID = "+str(userID)
    #cur1.execute("SELECT (%s) from PrefMaster where userID = (%s)",(emotion,int(userID)))
    cur.execute(query)
    data = cur.fetchone()
    print("Data",data)
    data=data[0]
    preference = data
    print("Preference : ",preference)
    songs=[]
    #Get user's preference from DB
    if preference == "Country" : 
        files = os.listdir(os.path.dirname("static/Music/Country/"))
        for musicFile in files:
            path = "static/Music/Country/"+musicFile    
            songs.append(path)
      
    
    elif preference == "Rap" :
        files = os.listdir(os.path.dirname("static/Music/Rap/"))
        for musicFile in files:
            path = "static/Music/Rap/"+musicFile    
            songs.append(path)
        
        
    elif preference == "R&B" : 
        files = os.listdir(os.path.dirname("static/Music/R&B/"))
        for musicFile in files:
            path = "static/Music/R&B/"+musicFile    
            songs.append(path)
        
           
    elif preference == "Rock" : 
        files = os.listdir(os.path.dirname("static/Music/Rock/"))
        for musicFile in files:
            path = "static/Music/Rock/"+musicFile    
            songs.append(path)
       
    mysql.connection.commit()
    cur.close()
    
    random.shuffle(songs)
           
    return render_template('MusicPlayer.html',playList=songs,loggedUser = user)
    
# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
    
