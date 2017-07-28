import json
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


class SpotifyHelper:
    def __init__(self):
        self.pl = None
        self.client_credentials_manager = SpotifyClientCredentials(client_id='dd4a138913544c12bf60424687fbbb83', client_secret='6d06781d13204cfca63f0f338a09a937')
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)

    def get_lyrics(self,song,artist):
        """ Given a song and artist combination, create the url from musixmatch """
        return "http://api.musixmatch.com/ws/1.1/matcher.lyrics.get?q_track="+song+"&q_artist="+artist+"&apikey=15520e2ca69218fca90985897c14f4f3"

    def extract_lyrics(self,lyric_url):
        """ Given the url from musixmatch for a selected artist/song combination, return lyrics """
        dob = requests.request("GET", lyric_url)
        dob_obj = json.loads(dob.content.decode("UTF-8"))
        if dob_obj["message"]["body"]:
            return ' '.join(dob_obj["message"]["body"]["lyrics"]["lyrics_body"].split("\n")[:-4])
        else:
            return None

    def parse_playlist_url(self,spidurl):
        """ Example spidurl: spotify:user:spotify:playlist:37i9dQZF1DWSkMjlBZAZ07  """
        tokens = spidurl.split(":")
        return {"user": tokens[2], "playlist_id": tokens[-1]}

    def get_tracks_from_playlist(self,spurl):
        # Given a playlist, I need to store
        # commonArtistName, commonSongName, SpotifyArtistURI, SpotifySongURI
        playlist_tracks = self.sp.user_playlist_tracks(**self.parse_playlist_url(spurl))
        tracks = playlist_tracks['items']
        while playlist_tracks['next']:
            playlist_tracks = self.sp.next(playlist_tracks)
            tracks.extend(playlist_tracks['items'])
            for i,v in enumerate(playlist_tracks['items']):
                yield { "SpotifyArtistURI": v['track']['artists'][0]['uri'],
                        "commonArtistName": v['track']['artists'][0]['name'],
                        "SpotifySongURI": v['track']['uri'],
                        "commonSongName": v['track']['name'],
                        "lyrics" : self.extract_lyrics(self.get_lyrics(v['track']['name'], v['track']['artists'][0]['name']))}


    def get_audio_analysis_for_track(self,trackids):
        """ Given a set of trackIDs, return the audio analysis generator """
        for uri in trackids:
            audio_features = self.sp.audio_features(uri)
            yield audio_features


if __name__ == "__main__":
    print("Spotify Helper")
    SPhelper = SpotifyHelper()
    kit = SPhelper.get_tracks_from_playlist("spotify:user:andreaskarsten:playlist:6wz8ygUKjoHfbU7tB9djeS")

    for k in kit:
        print(k)