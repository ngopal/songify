from flask import render_template, request, send_from_directory
from flaskexample import app
from sqlalchemy import create_engine
import psycopg2
from werkzeug.utils import secure_filename
import sys, os
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
from Models import DocModel
from VideoProcessing import VideoProc
import logging
import coloredlogs
coloredlogs.install()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

UPLOAD_FOLDER = './flaskapp/flaskexample/uploads/'
ALLOWED_EXTENSIONS = set(['.mp4'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

user = 'nikhilgopal' #add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'spotify_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

video_words_cache = {}

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
    songs = []
    return render_template("app.html", songs = songs[1:5])


@app.route('/result', methods = ['POST'])
def result_page():
    model_choices = {
        "validation" : "5LCI3ja6TCrmoQDLZ2FYem",
        "80s" : "19PgP2QSGPcm6Ve8VhbtpG",
        "90s" : "6wz8ygUKjoHfbU7tB9djeS",
        "insight" : "spotifyplaylistid",
        "muzic" : "7eHApqa9YVkuO6gELsju2j",
        "megan" : "4tZSI7b1rnGVMdkGeIbCI4"
    }
    # algo_choice = {
    #     "wordvec" : None,
    #     "lda" : None,
    #     "keamns-svd" : None,
    #     "svd-kmeans" : None
    # }
    if request.method == 'POST':
        result = request.form
        files = request.files
        print(result.getlist('musicdrop'))
        print(files)
        # words = ['n04162706 seat belt, seatbelt', 'n02965783 car mirror', 'n03452741 grand piano, grand', 'n04070727 refrigerator, icebox', 'n04162706 seat belt, seatbelt', 'n04200800 shoe shop, shoe-shop, shoe store', 'n03670208 limousine, limo', 'n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier', 'n04356056 sunglasses, dark glasses, shades', 'n04356056 sunglasses, dark glasses, shades', 'n03534580 hoopskirt, crinoline', 'n03032252 cinema, movie theater, movie theatre, movie house, picture palace']
        words = []

        # If Spotify URL provided then bleh
        # If dropdown, then blah
        # Else fail so hard it will make you cry

        for f in files:
            # print(files[f])
            # print(files[f].filename)
            # Check to see that vid_dir doesn't equal app.config['UPLOAD_FOLDER']
            # Create directories and upload files if it doesn't already exist
            vid_dir = os.path.join(app.config['UPLOAD_FOLDER'], files[f].filename.split(".")[0])
            frames_dir = os.path.join(app.config['UPLOAD_FOLDER'], files[f].filename.split(".")[0], "frames/")
            vid_file = os.path.join(vid_dir, files[f].filename)
            print(vid_file)
            print(video_words_cache)
            if vid_file not in video_words_cache:
                if not os.path.isdir(vid_dir):
                    os.mkdir(vid_dir)
                if not os.path.isdir(frames_dir):
                    os.mkdir(frames_dir)
                if not os.path.isfile(vid_file):
                    files[f].save(os.path.join(vid_dir, files[f].filename))

                # Downsample video frames into images
                # VP = VideoProc()
                # VP.downSampleVideo(vid_file, frames_dir, 60)
                #
                # # Run pre-trained NN on images
                # words = VP.analyzeImagesInDir(frames_dir)
                #
                # # Cache
                # video_words_cache[vid_file] = words
            else:
                words = video_words_cache[vid_file]

        # Model = DocModel(model_choices[result.getlist('musicdrop')[0]])
        print("TABLE CHOICE", "\""+model_choices[result.getlist('musicdrop')[0]]+"\"")
        Model = DocModel(model_choices[result.getlist('musicdrop')[0]])
        ####### DEBUGGING AND VALIDATION PURPOSE
        #
        words = ['car']
        #
        ###################

        # modelResults contains a dictionary with keys "uris" and "songs", each of which contains a sorted list
        # nullModel function takes a list of words as input
        algo_choice = result.getlist('algorithm')[0]
        if algo_choice == "wordvec":
            print("Running Word2Vec")
            modelResults = Model.nullModel(words)
        elif algo_choice == "kmeans-svd":
            print("Running Kmeans -> SVD")
            modelResults = Model.kmeans_svd(words)
        elif algo_choice == "svd-kmeans":
            print("Running SVD -> Kmeans")
            modelResults = Model.svd_kmeans(words)
        elif algo_choice == "lda":
            print("Running LDA")
            modelResults = Model.lda(words)
        else:
            print("Running Word2Vec")
            modelResults = Model.nullModel(words)

        songs = modelResults['songs']
        uris = modelResults["uris"]
        # spotify_track_url = "https://open.spotify.com/embed?uri=spotify:track:7LFer4drCtWSyD8oxORZtC&theme=white"
        logging.log(logging.INFO, songs)
        ## for debug
        # for song, score in songs:
        #     print(song, '\t', score)
        for i, v in enumerate(songs):
            print(v)
        ##
        spotify_track_url = "https://open.spotify.com/embed?uri="+uris[0][0]+"&theme=white"
        video_name = files[f].filename.split(".")[0]+'/'+files[f].filename.split(".")[0]+".mp4"
        return render_template("results.html", songs = songs[0:4], keywords=words, spotify_track_url=spotify_track_url, video_name=video_name)

@app.route("/video/<path:path>")
def simpler(path):
    return send_from_directory('uploads/', path)

@app.route("/video")
def simple():
    return send_from_directory('uploads/', "despacito/despacito.mp4")

@app.route("/simple.png")
def asda():
    import datetime
    from io import BytesIO
    import random

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    fig=Figure()
    ax=fig.add_subplot(111)
    x=[]
    y=[]
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    for i in range(10):
        x.append(now)
        now+=delta
        y.append(random.randint(0, 1000))
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

