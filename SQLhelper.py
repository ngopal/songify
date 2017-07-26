import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import json

class SQL:
    """Parse data from APIs into SQL"""
    def __init__(self):
        self.user = 'nikhilgopal' #add your username here (same as previous postgreSQL)
        self.host = 'localhost'
        self.dbname = 'spotify_db'
        self.engine = create_engine('postgres://%s@%s/%s'%(self.user,self.host,self.dbname))
        if not database_exists(self.engine.url):
            create_database(self.engine.url)

        self.con = None
        self.con = psycopg2.connect(database = self.dbname, user = self.user, host=self.host)
        print("Connected!")
        self.con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        self.cur = self.con.cursor()


    def createDatabase(self):
        con = psycopg2.connect(user = user, host=host)
        cur.execute("CREATE DATABASE %s  ;" % self.dbname)
        print("Created DB", self.dbname)
        # create schema from definition

    def createTable(self, schemaCommand):
        self.cur.execute(schemaCommand)

    def dropTable(self, tableName):
        self.cur.execute("DROP TABLE "+tableName)

    def constructSchema(self, schema):
        schemaCommand = 'CREATE TABLE '
        for k, v in schema.items():
            if k == "tablename":
                schemaCommand += v+" (\n"
            else:
                schemaCommand += "  "+k+" "+v+",\n"
        schemaCommand = schemaCommand[:-2]
        schemaCommand += "\n)"
        print(schemaCommand)
        return schemaCommand

    def insertRowIntoTable(self,table,row_data):
        #command = "INSERT INTO vendors(vendor_name) VALUES(%s) RETURNING vendor_id;"
        # "INSERT INTO items (info, city, price) VALUES (%s, %s, %s);"
        insert_command = "INSERT INTO " + table +"( "
        for k in row_data:
            insert_command += k+", "
        insert_command = insert_command[:-2]
        insert_command += ") VALUES ( "
        for i,k in enumerate(row_data):
            insert_command += "%s, "
        # for k,v in row_data.items():
        #     insert_command += json.dumps(v)+", "
        insert_command = insert_command[:-2]
        insert_command += " )"
        print(insert_command)
        #print(tuple(row_data.values()))
        #self.cur.execute(insert_command, tuple(row_data.values()))
        self.cur.execute("""INSERT INTO spotifyplaylistid( SpotifyArtistURI, commonArtistName, SpotifySongURI, commonSongName, lyrics) VALUES ( %s, %s, %s, %s, %s )""", ("a", "3", "b", "c", "s"))
        return

    def customCommand(self, format, vals):
        self.cur.execute(format, vals)
        return


    def pullPlaylist(self, spotifyPlaylistID, generator):
        """Given a spotify playlist ID and python generator of columns, insert the data into the table"""
        # check if table already exists
        pass

if __name__ == "__main__":
    helper = SQL()

    ###### EXAMPLE TO CREATE TABLE #########
    # schema = {
    #     "tablename" : "spotifyplaylistID",
    #     "index" : "SERIAL PRIMARY KEY",
    #     "commonSongName" : "VARCHAR",
    #     "commonArtistName" : "VARCHAR",
    #     "SpotifySongURI" : "VARCHAR",
    #     "SpotifyArtistURI" : "VARCHAR",
    #     "lyrics" : "VARCHAR"
    # }
    # sc = helper.constructSchema(schema)
    #helper.createTable(sc)
    ######################

    # WARNING: UNCOMMENTING CODE BELOW WILL DROP THE TABLE
    ##
    ##
    ##
    #
    helper.dropTable("spotifyplaylistid")
    #
    ##
    ##########################



    #### EXAMPLE TO INSERT DATA INTO TABLE AND PRINT IT
    #
    #row = {'SpotifyArtistURI': 'spotify:artist:2ht3wxeT69CzyKFChNnNAB', 'commonArtistName': 'Big Boi', 'SpotifySongURI': 'spotify:track:4i2HYTGM7yHty0qVhD3lqD', 'commonSongName': 'Kryptonite - feat. Big Boi', 'lyrics': 'I be on it all night, man I be on it (day day) All day, straight up pimp If you want me, you can find me in the A!  AYE!(I\'m on it) Time and time again, I gotta turn back round and tell these hoes That I am the H-N-I-C, bitch that\'s just the way it goes I be on that shit that\'ll have you on that "I don\'t want no mo\'" At this time, I need all my freak hoes to get down on the flo\' If you came to rep your set, right now nigga, let \'em know If it\'s jail I get for stompin\' a hater to sleep, fuck it, I go Freak, I\'ll be off in the wheep Straight geeked swerving down your street In a stolen Bonneville with 23\'s on the feet The Legend, Rocky D Brown, back in town to plea you down Give me face, I love the sound Slap the taste, they hit the ground Back in the A Cliqued up, picked up with some people that don\'t play On that Kryptonite stay So high, we might fly awwwaaayyy  I be on that Kryptonite Straight up on that Kryptonite'}
    #row2 = {'SpotifyArtistURI': 'spotify:artist:adsadadasds', 'commonArtistName': 'asda Boi', 'SpotifySongURI': 'spotify:track:asdsadsdad', 'commonSongName': 'Kryptonite - feat. Big asd', 'lyrics': 'I be on it all adsa, man I be on it (day day) All day, straight up pimp If you want me, you can find me in the A!  AYE!(I\'m on it) Time and time again, I gotta turn back round and tell these hoes That I am the H-N-I-C, bitch that\'s just the way it goes I be on that shit that\'ll have you on that "I don\'t want no mo\'" At this time, I need all my freak hoes to get down on the flo\' If you came to rep your set, right now nigga, let \'em know If it\'s jail I get for stompin\' a hater to sleep, fuck it, I go Freak, I\'ll be off in the wheep Straight geeked swerving down your street In a stolen Bonneville with 23\'s on the feet The Legend, Rocky D Brown, back in town to plea you down Give me face, I love the sound Slap the taste, they hit the ground Back in the A Cliqued up, picked up with some people that don\'t play On that Kryptonite stay So high, we might fly awwwaaayyy  I be on that Kryptonite Straight up on that Kryptonite'}
    #rows = { str(r): {'SpotifyArtistURI': 'spotify:artist:'+str(r), 'commonArtistName': str(r), 'SpotifySongURI': 'spotify:track:'+str(r), 'commonSongName': 'BIG '+str(r), 'lyrics': 'jhksadhahdkjshdkahhsjdas'} for r in range(100)}
    #dfA = pd.DataFrame().from_dict(rows, orient='index')
    #print(dfA)
    #dfA.to_sql('spotifyplaylistid', helper.engine, if_exists='append')

    #print(pd.read_sql_table('spotifyplaylistid', helper.engine))



