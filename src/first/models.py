from django.db import models


class FileModel(models.Model):
    file = models.FileField(upload_to='file/')
    llm_output = models.JSONField(null=True, blank=True)

class ImageModel(models.Model):
    file = models.ImageField(upload_to='img/')
    llm_output = models.JSONField(null=True, blank=True)