import pandas as pd
import sys
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
from Helpers.SQLhelper import SQL
from Helpers.Spotifyhelper import SpotifyHelper

# Initialize Spotify helper
#pl = "spotify:user:acesamped:playlist:1VvcEiPSvXOBX9HYkiP0IL" ## Insight_TestDataSet
#pl = "spotify:user:thesoundsofspotify:playlist:4SMubSJhL8oHG1RNa6RGkQ" ## Sound of Seattle Playlist
#pl = "spotify:user:andreaskarsten:playlist:6wz8ygUKjoHfbU7tB9djeS" ## 90's West Coast G-Funk
# pl = "spotify:user:myplay.com:playlist:19PgP2QSGPcm6Ve8VhbtpG" # 80's Smash Hits
# pl = "spotify:user:acesamped:playlist:7eHApqa9YVkuO6gELsju2j" # muzic
# pl = "spotify:user:spotifycharts:playlist:37i9dQZEVXbLRQDuF5jeBp" # USA Top 50
pl = "spotify:user:mmegaard:playlist:4tZSI7b1rnGVMdkGeIbCI4"
# pl = "spotify:user:spotify:playlist:37i9dQZF1DWSkMjlBZAZ07" #happy folk
# pl = "spotify:user:napstersean:playlist:3vxotOnOGDlZXyzJPLFnm2" #Hipster International
# pl = "spotify:user:myplay.com:playlist:68bXT1MZWZvLOJc0FZrgf7" #Dance for days
pl_name = pl.split(":")[-1]
print(pl_name)
SPhelper = SpotifyHelper()

# Connect to Database
DBhelper = SQL()

# Can drop table using code below
# DBhelper.dropTable(pl_name)

# Create Generator
artist_lyric_generator = (k for k in SPhelper.get_tracks_from_playlist(pl))
index_num = 0
for a in SPhelper.get_tracks_from_playlist(pl):
    row = { str(index_num) : a}
    print(row)
    try:
        dfA = pd.DataFrame().from_dict(row, orient='index')
        dfA.to_sql(pl_name, DBhelper.engine, if_exists='append')
        # print(a)
        index_num += 1
    except:
        pass


# Print dataframe
#print(pd.read_sql_table(pl_name, DBhelper.engine))