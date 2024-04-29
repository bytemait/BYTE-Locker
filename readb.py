import sqlite3

conn = sqlite3.connect('database.db')
c = conn.cursor()
# Ask the user to enter the command and print the output and keep on doing this until the user enters 'exit'
c.execute('SELECT person_id, camera_id, timestamp FROM detections')
print(c.fetchall())
conn.close()