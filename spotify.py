import pandas as pd
import json
import spotipy, requests
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import sqlalchemy
import operator
import spacy

def get_lyrics(song,artist):
    return "http://api.musixmatch.com/ws/1.1/matcher.lyrics.get?q_track="+song+"&q_artist="+artist+"&apikey=15520e2ca69218fca90985897c14f4f3"

def extract_lyrics(lyric_url):
    dob = requests.request("GET", lyric_url)
    dob_obj = json.loads(dob.content.decode("UTF-8"))
    if dob_obj["message"]["body"]:
        return dob_obj["message"]["body"]["lyrics"]["lyrics_body"]
    else:
        return None

def parse_playlist_url(spidurl):
    tokens = spidurl.split(":")
    return {"user": tokens[2], "playlist_id": tokens[-1]}

def get_tracks_from_playlist(spurl, getLyrics = False):
    # Given a playlist, I need to store
    # commonArtistName, commonSongName, SpotifyArtistURI, SpotifySongURI
    playlist_tracks = sp.user_playlist_tracks(**parse_playlist_url(spurl))
    for i,v in enumerate(playlist_tracks['items']):
        yield { "SpotifyArtistURI": v['track']['artists'][0]['uri'],
                "commonArtistName": v['track']['artists'][0]['name'],
                "SpotifySongURI": v['track']['uri'],
                "commonSongName": v['track']['name']}
        if getLyrics:
            yield { "SpotifyArtistURI": v['track']['artists'][0]['uri'],
                    "commonArtistName": v['track']['artists'][0]['name'],
                    "SpotifySongURI": v['track']['uri'],
                    "commonSongName": v['track']['name'],
                    "lyrics" : extract_lyrics(get_lyrics(v['track']['name'], v['track']['artists'][0]['name']))}


client_credentials_manager = SpotifyClientCredentials(client_id='dd4a138913544c12bf60424687fbbb83', client_secret='6d06781d13204cfca63f0f338a09a937')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

pl = "spotify:user:acesamped:playlist:1VvcEiPSvXOBX9HYkiP0IL"

artist_lyric_generator = (k for k in get_tracks_from_playlist(pl, getLyrics=True))
# I will want to use this generator to get data into a MySQL database
df = pd.DataFrame(artist_lyric_generator)
df.to_pickle("./artist_lyrics.p")
#df = pd.read_pickle("./artist_lyrics.p")
df.dropna(inplace=True)
#print(df.head())

# Process text
# pd.options.display.max_colwidth = 1000
df[["lyrics"]] = df[["lyrics"]].applymap(lambda x: x.split('\n')[:-4])

doc_models = []
nlp = spacy.load('en')
for r in df.iterrows():
    doc_models.append({"song": r[1]["commonSongName"], "model" : nlp(''.join(r[1]["lyrics"]))})

# in_str = "beer in bar" #good example
# in_str = "Anyone up for a road trip?"
# in_str = "living room"
# real_test = ['n03041632 cleaver, meat cleaver, chopper', 'n04332243 strainer', "n03954731 plane, carpenter's plane, woodworking plane", 'n03207941 dishwasher, dish washer, dishwashing machine', 'n03729826 matchstick']
real_test = ['n04162706 seat belt, seatbelt', 'n02965783 car mirror', 'n03452741 grand piano, grand', 'n04070727 refrigerator, icebox', 'n04162706 seat belt, seatbelt', 'n04200800 shoe shop, shoe-shop, shoe store', 'n03670208 limousine, limo', 'n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier', 'n04356056 sunglasses, dark glasses, shades', 'n04356056 sunglasses, dark glasses, shades', 'n03534580 hoopskirt, crinoline', 'n03032252 cinema, movie theater, movie theatre, movie house, picture palace']
reat_test = [i.split(" ")[-1] for i in real_test]
in_str = ''.join(real_test)
inModel = nlp(in_str)
results = {}
for i,d in enumerate(doc_models):
    simScore = inModel.similarity(d["model"])
    results[d["song"]] = simScore

for k, v in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
    print(k, v)