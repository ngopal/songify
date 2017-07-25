import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

class SQL:
    """Parse data from APIs into SQL"""
    def __init__(self):
        user = 'nikhilgopal' #add your username here (same as previous postgreSQL)
        host = 'localhost'
        dbname = 'spotify_db'
        db = create_engine('postgres://%s%s/%s'%(user,host,dbname), encoding='utf-8', echo=True, case_sensitive=True)
        con = None
        con = psycopg2.connect(database = dbname, user = user, host=host)
        print("Connected!")
        #con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        #cur = con.cursor()

    def createDatabase(self, schema):
        con = psycopg2.connect(user = user, host=host)
        cur.execute("CREATE DATABASE %s  ;" % self.dbname)
        print("Created DB", self.dbname)
        # create schema from definition

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


    def pullPlaylist(self, spotifyPlaylistID, generator):
        """Given a spotify playlist ID and python generator of columns, insert the data into the table"""
        # check if table already exists
        pass

if __name__ == "__main__":
    helper = SQL()
    schema = {
        "tablename" : "spotifyplaylistID",
        "entryId" : "SERIAL PRIMARY KEY",
        "commonSongName" : "VARCHAR(255)",
        "commonArtistName" : "VARCHAR(255)",
        "spotifySongId" : "VARCHAR(255)",
        "spotifyArtistId" : "VARCHAR(255)"
    }
    helper.constructSchema(schema)


