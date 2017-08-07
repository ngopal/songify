import json
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np


class SpotifyHelper:
    def __init__(self):
        self.pl = None
        self.client_credentials_manager = SpotifyClientCredentials(client_id='dd4a138913544c12bf60424687fbbb83', client_secret='6d06781d13204cfca63f0f338a09a937')
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        self.user = self.sp.user('acesamped')
        self.userPlaylist = 'spotify:user:acesamped:playlist:1PB7C1ACj6d9ZKQh2U1SrU'

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

        # Go through first page of tracks results
        playlist_tracks = self.sp.user_playlist_tracks(**self.parse_playlist_url(spurl))
        tracks = playlist_tracks['items']
        for i,v in enumerate(playlist_tracks['items']):
            yield { "SpotifyArtistURI": v['track']['artists'][0]['uri'],
                    "commonArtistName": v['track']['artists'][0]['name'],
                    "SpotifySongURI": v['track']['uri'],
                    "commonSongName": v['track']['name'],
                    "lyrics" : self.extract_lyrics(self.get_lyrics(v['track']['name'], v['track']['artists'][0]['name']))}

        # If there are additional results (past 100 values), then continue yielding...
        while playlist_tracks['next']:
            playlist_tracks = self.sp.next(playlist_tracks)
            print(playlist_tracks)
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

    def get_genre_given_song(self, song_sp_url):
        song_data = self.sp.artist(song_sp_url)
        if "genres" in song_data:
            for g in song_data["genres"]:
                yield { "SpotifyArtistURI" : song_data["uri"], "Genre" : g}

    def clear_web_playlist(self):
        """ Expecting no input, and a return value of None """
        userPlaylist = self.userPlaylist

        # Get all tracks in playlist, store in list
        # self.sp.user_playlist("acesamped", playlist_id=self.userPlaylist)
        tracks_to_delete = ["spotify:track:7lcCLXghOHf8xERQ9BkS3n"]

        # Pass tracks to delete to function below
        self.sp.user_playlist_remove_all_occurrences_of_tracks("acesamped", userPlaylist, tracks_to_delete)
        return

    def add_songs_to_user_playlist(self, songs_to_add):
        """ Expecting a list of spotify track URIs """
        # https://stackoverflow.com/questions/11512412/adding-a-song-to-a-playlist-with-pyspotify
        userPlaylist = self.userPlaylist

        # Add songs to playlist
        self.sp.user_playlist_add_tracks("acesamped", userPlaylist, songs_to_add)
        return

    def get_tracks_without_lyrics(self, spurl):
        # Go through first page of tracks results
        playlist_tracks = self.sp.user_playlist_tracks(**self.parse_playlist_url(spurl))
        tracks = playlist_tracks['items']
        for i,v in enumerate(playlist_tracks['items']):
            yield { "SpotifyArtistURI": v['track']['artists'][0]['uri'],
                    "commonArtistName": v['track']['artists'][0]['name'],
                    "SpotifySongURI": v['track']['uri'],
                    "commonSongName": v['track']['name']}

        # If there are additional results (past 100 values), then continue yielding...
        while playlist_tracks['next']:
            playlist_tracks = self.sp.next(playlist_tracks)
            # print(playlist_tracks)
            tracks.extend(playlist_tracks['items'])
            for i,v in enumerate(playlist_tracks['items']):
                yield { "SpotifyArtistURI": v['track']['artists'][0]['uri'],
                        "commonArtistName": v['track']['artists'][0]['name'],
                        "SpotifySongURI": v['track']['uri'],
                        "commonSongName": v['track']['name']}
        return

    def randomly_select_n_songs(self, n_songs):
        """ Randomly select n_songs from my 'muzic' playlist
            Returns a list of spotify TrackURIs
        """
        # playlist_source = "spotify:user:acesamped:playlist:7eHApqa9YVkuO6gELsju2j"
        playlist_source = "spotify:user:myplay.com:playlist:19PgP2QSGPcm6Ve8VhbtpG"
        track_generator = self.get_tracks_without_lyrics(playlist_source)
        # random selection without replacement
        # reuse get_tracks_from_playlist function, but randomly select numbers from 0 to len(tracks)
        # print(tracks)
        # print(len(tracks))
        # indx = [np.random.randint(0,len(tracks)) for i in range(n_songs)]
        songs = list(track_generator)
        indx = set(np.random.choice(len(songs), n_songs))
        print(indx)

        for index, track in enumerate(songs):
            if index in indx:
                # print(index, track)
                print(track['commonArtistName'], '\t', track['commonSongName'])
            else:
                pass

        return


if __name__ == "__main__":
    print("Spotify Helper")
    SPhelper = SpotifyHelper()
    # kit = SPhelper.get_tracks_from_playlist("spotify:user:andreaskarsten:playlist:6wz8ygUKjoHfbU7tB9djeS")
    #
    # for k in kit:
    #     print(k)
    # SPhelper.clear_web_playlist()
    SPhelper.randomly_select_n_songs(20)