from SQLhelper import SQL
from Spotifyhelper import SpotifyHelper
import pandas as pd

# Initialize Spotify helper
#pl = "spotify:user:acesamped:playlist:1VvcEiPSvXOBX9HYkiP0IL" ## Insight_TestDataSet
#pl = "spotify:user:thesoundsofspotify:playlist:4SMubSJhL8oHG1RNa6RGkQ" ## Sound of Seattle Playlist
#pl = "spotify:user:andreaskarsten:playlist:6wz8ygUKjoHfbU7tB9djeS" ## 90's West Coast G-Funk
pl = "spotify:user:myplay.com:playlist:19PgP2QSGPcm6Ve8VhbtpG" # 80's Smash Hits
pl_name = pl.split(":")[-1]
print(pl_name)
#pl_name = "spotifyplaylistid"
SPhelper = SpotifyHelper(pl)

# Connect to Database
DBhelper = SQL()

# Can drop table using code below
#DBhelper.dropTable("4SMubSJhL8oHG1RNa6RGkQ")

# Create Generator
# artist_lyric_generator = (k for k in SPhelper.get_tracks_from_playlist(pl))
index_num = 0
for a in SPhelper.get_tracks_from_playlist(pl):
    row = { str(index_num) : a}
    print(row)
    dfA = pd.DataFrame().from_dict(row, orient='index')
    dfA.to_sql(pl_name, DBhelper.engine, if_exists='append')
    print(a)
    index_num += 1


# Print dataframe
#print(pd.read_sql_table(pl_name, DBhelper.engine))