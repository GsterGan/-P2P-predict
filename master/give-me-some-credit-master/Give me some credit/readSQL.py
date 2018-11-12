import pymysql
db =    pymysql.connect("localhost","root","123456","debt")
cursor = db.cursor()
cursor.execute("select * from information where age<23")
data = cursor.fetchall()
print(data)
db.close()
