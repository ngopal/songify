import sys
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
import pandas as pd

from Helpers.SQLhelper import SQL
from Helpers.Spotifyhelper import SpotifyHelper

# Initialize Spotifyhelper
SPhelper = SpotifyHelper()

# Connect to given DB
DBhelper = SQL()

# DBhelper.dropTable("aa_"+pl_name)

# Extract track ids
all_tables = ["19PgP2QSGPcm6Ve8VhbtpG", "37i9dQZF1DWTJ7xPn4vNaz","37i9dQZF1DX1ewVhAJ17m4","37i9dQZF1DX5bjCEbRU4SJ","37i9dQZF1DXbTxeAdrVG2l","37i9dQZF1DXcBWIGoYBM5M","3vxotOnOGDlZXyzJPLFnm2","49oW3sCI91kB2YGw7hsbBv","4tZSI7b1rnGVMdkGeIbCI4","68bXT1MZWZvLOJc0FZrgf7","6wz8ygUKjoHfbU7tB9djeS","7eHApqa9YVkuO6gELsju2j","spotifyplaylistid", "3nrwJoFbrMKSGeHAxaoYSC", "37i9dQZF1DX1XDyq5cTk95"]
frames = []
for p in all_tables:
    pdf = pd.read_sql_query("SELECT * FROM \"" + p +"\"", DBhelper.engine)
    frames.append(pdf)

pdf = pd.concat(frames)


# Get all unique artist names
# index_num = 0
# for a in pdf.SpotifyArtistURI.unique():
#     if a:
#         art_gen = SPhelper.get_genre_given_song(a)
#         for g in art_gen:
#             dfA = pd.DataFrame().from_dict({ str(index_num) : g }, orient='index')
#             # print(dfA)
#             dfA.to_sql("artist_genre", DBhelper.engine, if_exists='append')
#             index_num += 1

# print(dfA)

# index_num = 0
# for p in pdf.itertuples():
#     artist_uri = p[1]
#     dfA = pd.DataFrame().from_dict({ str(index_num) : artist_uri }, orient='index')
#     print(dfA)
#     dfA.to_sql("ag_"+pl_name, DBhelper.engine, if_exists='replace')
#     index_num += 1


print(pd.read_sql_table("artist_genre", DBhelper.engine))

