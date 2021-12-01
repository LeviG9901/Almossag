import mysql.connector
from datetime import datetime

# Adatbázishoz kapcsolódás
db = mysql.connector.connect(
    user='root', password='rootpws', # Felhasználónév, jelszó megadása
    host='localhost', database='vezetes', # Host, és az adatbázis nevének megadása
    auth_plugin='mysql_native_password' # Hitelesítési plugin
)

cursor = db.cursor()
# Táblába új sor beillesztése
def insert_into_table(sql,values): # SQL kód, és az értékek megadása paraméterként
    cursor.execute(sql, values) # SQL kód lefuttatása a "cursor" segítségével
    db.commit() # A "commit" után történik az SQL kód lefuttatása
# Táblába sorok törlésére
def delete_data(sql):  # SQL kód megadása paraméterként
    cursor.execute(sql)
    db.commit()