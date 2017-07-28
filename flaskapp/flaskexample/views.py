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
        "80s" : "19PgP2QSGPcm6Ve8VhbtpG",
        "90s" : "6wz8ygUKjoHfbU7tB9djeS",
        "insight" : "spotifyplaylistid",
        "muzic" : "7eHApqa9YVkuO6gELsju2j",
        "megan" : "4tZSI7b1rnGVMdkGeIbCI4"
    }
    if request.method == 'POST':
        result = request.form
        files = request.files
        print(result.getlist('musicdrop'))
        print(files)
        # words = ['n04162706 seat belt, seatbelt', 'n02965783 car mirror', 'n03452741 grand piano, grand', 'n04070727 refrigerator, icebox', 'n04162706 seat belt, seatbelt', 'n04200800 shoe shop, shoe-shop, shoe store', 'n03670208 limousine, limo', 'n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier', 'n04356056 sunglasses, dark glasses, shades', 'n04356056 sunglasses, dark glasses, shades', 'n03534580 hoopskirt, crinoline', 'n03032252 cinema, movie theater, movie theatre, movie house, picture palace']
        words = []
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
                VP = VideoProc()
                VP.downSampleVideo(vid_file, frames_dir, 60)

                # Run pre-trained NN on images
                words = VP.analyzeImagesInDir(frames_dir)

                # Cache
                video_words_cache[vid_file] = words
            else:
                words = video_words_cache[vid_file]

        Model = DocModel(model_choices[result.getlist('musicdrop')[0]])
        songs = Model.nullModel(words)
        return render_template("app.html", songs = songs[1:5], keywords=words)


@app.route("/simple.png")
def simple():
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

