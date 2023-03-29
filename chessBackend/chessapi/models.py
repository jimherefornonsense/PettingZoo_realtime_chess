from django.db import models

# Create your models here.
class BoardInstance(models.Model):
    uuid = models.CharField(max_length=1000)
    time = models.IntegerField()
    board = models.CharField(max_length=500)

