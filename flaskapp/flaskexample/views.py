from flask import render_template, request
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

import sys
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
from Models import DocModel

user = 'nikhilgopal' #add your username here (same as previous postgreSQL)
host = 'localhost'
dbname = 'spotify_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

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
        print(result.getlist('musicdrop'))
        Model = DocModel(model_choices[result.getlist('musicdrop')[0]])
        keyws = ['n04162706 seat belt, seatbelt', 'n02965783 car mirror', 'n03452741 grand piano, grand', 'n04070727 refrigerator, icebox', 'n04162706 seat belt, seatbelt', 'n04200800 shoe shop, shoe-shop, shoe store', 'n03670208 limousine, limo', 'n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier', 'n04356056 sunglasses, dark glasses, shades', 'n04356056 sunglasses, dark glasses, shades', 'n03534580 hoopskirt, crinoline', 'n03032252 cinema, movie theater, movie theatre, movie house, picture palace']
        songs = Model.nullModel(keyws)
        return render_template("app.html", songs = songs[1:5])