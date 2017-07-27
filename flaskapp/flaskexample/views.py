from flask import render_template, request
from flaskexample import app
from sqlalchemy import create_engine
import psycopg2
from werkzeug.utils import secure_filename
import sys, os
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
from Models import DocModel
from VideoProcessing import VideoProc

UPLOAD_FOLDER = './flaskapp/flaskexample/uploads/'
ALLOWED_EXTENSIONS = set(['.mp4'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

user = 'nikhilgopal' #add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'spotify_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
                           title = 'Home', user = { 'nickname': 'Miguel' },
                           )

# @app.route('/db')
# def birth_page():
#     sql_query = """
#                 SELECT * FROM "19PgP2QSGPcm6Ve8VhbtpG";
#                 """
#     query_results = pd.read_sql_query(sql_query,con)
#     print(query_results)
#     births = ""
#     for i in range(0,10):
#         births += query_results.iloc[i]['commonSongName']
#         births += "<br>"
#     print(births)
#     return births

@app.route('/app')
def app_page():
    songs = [(i,i) for i in range(10)]
    return render_template("app.html", songs = songs[1:5])


@app.route('/result', methods = ['POST'])
def result_page():
    model_choices = {
        "80s" : "19PgP2QSGPcm6Ve8VhbtpG",
        "90s" : "6wz8ygUKjoHfbU7tB9djeS",
        "insight" : "spotifyplaylistid"
    }
    if request.method == 'POST':
        result = request.form
        files = request.files
        print(result.getlist('musicdrop'))
        print(files)
        words = []
        for f in files:
            print(files[f])
            print(files[f].filename)
            # Mkdirs and upload files
            os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], files[f].filename.split(".")[0]))
            os.mkdir(os.path.join(app.config['UPLOAD_FOLDER'], files[f].filename.split(".")[0], "frames"))
            files[f].save(os.path.join(app.config['UPLOAD_FOLDER'], files[f].filename.split(".")[0], files[f].filename))

            # Run image to frames
            VP = VideoProc()
            VP.downSampleVideo(os.path.join(app.config['UPLOAD_FOLDER'], files[f].filename.split(".")[0], files[f].filename), os.path.join(app.config['UPLOAD_FOLDER'], files[f].filename.split(".")[0], "frames"), 15)

            # Run NN on images
            words = VP.analyzeImagesInDir(os.path.join(app.config['UPLOAD_FOLDER'], files[f].filename.split(".")[0], "frames"))

        Model = DocModel(model_choices[result.getlist('musicdrop')[0]])
        # keyws = ['n04162706 seat belt, seatbelt', 'n02965783 car mirror', 'n03452741 grand piano, grand', 'n04070727 refrigerator, icebox', 'n04162706 seat belt, seatbelt', 'n04200800 shoe shop, shoe-shop, shoe store', 'n03670208 limousine, limo', 'n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier', 'n04356056 sunglasses, dark glasses, shades', 'n04356056 sunglasses, dark glasses, shades', 'n03534580 hoopskirt, crinoline', 'n03032252 cinema, movie theater, movie theatre, movie house, picture palace']
        # songs = Model.nullModel(keyws)
        songs = Model.nullModel(words)
        return render_template("app.html", songs = songs[1:5])