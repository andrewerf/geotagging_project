from peewee import *


user = 'geotagging_db_user'
password = '123456'
db_name = 'geotagging_db'
db_host = '212.24.111.5'

db_handler = MySQLDatabase(db_name,
						   user=user,
						   password = password,
						   host=db_host)
#db_handler = SqliteDatabase('/media/andrew/Data/Temp/nn/geotagging_db.sqlite3')

class BaseModel(Model):
	class Meta:
		database = db_handler

class Sight(BaseModel):
	id = PrimaryKeyField(null=False)
	area = CharField(max_length=30)
	type_id = IntegerField()
	db_table = "sights"

	class Meta:
		db_table = "sights"

class Descriptor(BaseModel):
	id = PrimaryKeyField(null=False)
	image_id = CharField(max_length=30, unique=True)
	descriptor = CharField(max_length=40000)
	sight_id = ForeignKeyField(Sight, backref='descrs')
	db_table = "descriptors"

	class Meta:
		db_table = "descriptors"

class Streets(BaseModel):
	id = PrimaryKeyField(null=False)
	pos = CharField(max_length=40, unique=True)
	house_number = CharField(max_length=50)
	street = CharField(max_length=15000)
	city = CharField(max_length=50)
	class Meta:
		db_table = "streets"

