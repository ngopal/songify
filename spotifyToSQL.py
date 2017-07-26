from SQLhelper import SQL
from Spotifyhelper import SpotifyHelper

# Initialize Spotify helper
pl = "spotify:user:acesamped:playlist:1VvcEiPSvXOBX9HYkiP0IL"
SPhelper = SpotifyHelper(pl)


# Create Generator
# artist_lyric_generator = (k for k in SPhelper.get_tracks_from_playlist(pl))
# for a in artist_lyric_generator:
#     print(a)

# Connect to Database
helper = SQL()
schema = {
    "tablename" : "1VvcEiPSvXOBX9HYkiP0IL",
    "entryId" : "SERIAL PRIMARY KEY",
    "commonSongName" : "VARCHAR",
    "commonArtistName" : "VARCHAR",
    "spotifySongId" : "VARCHAR",
    "spotifyArtistId" : "VARCHAR",
    "lyrics" : "VARCHAR"
}
sc = helper.constructSchema(schema)
#helper.dropTable("spotifyplaylistid")
helper.createTable(sc)
# Insert data into Database

