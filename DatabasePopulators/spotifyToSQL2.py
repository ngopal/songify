import sys
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
import pandas as pd

from Helpers.SQLhelper import SQL
from Helpers.Spotifyhelper import SpotifyHelper

# pl = "spotify:user:acesamped:playlist:1VvcEiPSvXOBX9HYkiP0IL" ## Insight_TestDataSet
# pl = "spotify:user:thesoundsofspotify:playlist:4SMubSJhL8oHG1RNa6RGkQ" ## Sound of Seattle Playlist
# pl = "spotify:user:andreaskarsten:playlist:6wz8ygUKjoHfbU7tB9djeS" ## 90's West Coast G-Funk
# pl = "spotify:user:myplay.com:playlist:19PgP2QSGPcm6Ve8VhbtpG" # 80's Smash Hits
# pl = "spotify:user:acesamped:playlist:7eHApqa9YVkuO6gELsju2j" # muzic


# pl_name = "6wz8ygUKjoHfbU7tB9djeS"
# pl_name = "19PgP2QSGPcm6Ve8VhbtpG"
pl_name = "7eHApqa9YVkuO6gELsju2j"

# Initialize Spotifyhelper
SPhelper = SpotifyHelper()

# Connect to given DB
DBhelper = SQL()

# DBhelper.dropTable("aa_"+pl_name)

# Extract track ids
pdf = pd.read_sql_query(" SELECT \"SpotifySongURI\" FROM \""+pl_name+"\" ", DBhelper.engine)

audio_features = SPhelper.get_audio_analysis_for_track(pdf["SpotifySongURI"])

index_num = 0
for af in audio_features:
    dfA = pd.DataFrame().from_dict({ str(index_num) : af[0] }, orient='index')
    print(dfA)
    dfA.to_sql("aa_"+pl_name, DBhelper.engine, if_exists='replace')
    index_num += 1


#print(pd.read_sql_table("aa"+"_6wz8ygUKjoHfbU7tB9djeS", DBhelper.engine))

