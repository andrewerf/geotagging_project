from peewee import *


user = 'geotagging_db_user'
password = '123456'
db_name = 'geotagging_db'
db_host = '212.24.111.5'

db_handler = MySQLDatabase(db_name,
						   user=user,
						   password = password,
						   host=db_host)

class BaseModel(Model):
	class Meta:
		database = db_handler


class Descriptor(BaseModel):
	id = PrimaryKeyField(null=False)
	image_id = IntegerField(null=False)
	descriptor = CharField(max_length=40000)
	sight_id = IntegerField()
	db_table = "descriptors"

	class Meta:
		db_table = "descriptors"

