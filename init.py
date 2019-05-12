
import sqlite3
import os
import shutil

con = sqlite3.connect('db.sqlite')
cur = con.cursor()

cur.executescript('''
Drop Table If Exists Result;
Create Table Result(
id Integer Primary Key,
model_file_path text unique,
Nz Integer,
Nlayer Integer,
Nh Integer,
Nbatch Integer,
Nitr Integer,
max_sink_itr Integer,
eps Reel
)
''')


if os.path.exists('./tmp'):
    shutil.rmtree('./tmp')
os.mkdir('./tmp')

