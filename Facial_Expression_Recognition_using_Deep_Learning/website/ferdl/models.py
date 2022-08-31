from django.db import models

# Create your models here.
from django.db import models

# Create your models here.

class Registrationtable(models.Model):


    objects = None
    username = models.CharField(max_length=20)
    password = models.CharField(max_length=10)
    mobile_no = models.CharField(max_length=11,unique=True)
    gender = models.CharField(max_length=10)
    register_time = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = "registrationtable"


class Feedback(models.Model):
    objects = None
    user_id = models.IntegerField()

    feedback = models.CharField(max_length=1000)
    class Meta:
        db_table = "feedback"

    

