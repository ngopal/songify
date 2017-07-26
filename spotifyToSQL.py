from SQLhelper import SQL
from Spotifyhelper import SpotifyHelper
import pandas as pd

# Initialize Spotify helper
pl = "spotify:user:acesamped:playlist:1VvcEiPSvXOBX9HYkiP0IL"
SPhelper = SpotifyHelper(pl)

# Connect to Database
helper = SQL()

# Create Generator
artist_lyric_generator = (k for k in SPhelper.get_tracks_from_playlist(pl))
index_num = 0
for a in artist_lyric_generator:
    row = { str(index_num) : a}
    dfA = pd.DataFrame().from_dict(row, orient='index')
    dfA.to_sql('spotifyplaylistid', helper.engine, if_exists='append')
    print(a)
    index_num += 1


# Print dataframe
print(pd.read_sql_table('spotifyplaylistid', helper.engine))