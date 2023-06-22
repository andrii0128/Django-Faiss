from django.db import models

# Create your models here.
class Utterance(models.Model):
    utterance = models.CharField(max_length=255)
    intent_category = models.CharField(max_length=255)
