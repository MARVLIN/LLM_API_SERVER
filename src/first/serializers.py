from dataclasses import field
from rest_framework import serializers

from .models import FileModel, ImageModel


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileModel
        fields = "__all__"


class MultipleFileSerializer(serializers.Serializer):
    file = serializers.ListField(
        child=serializers.FileField()
    )
    llm_output = serializers.JSONField(required=False, allow_null=True)


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageModel
        fields = "__all__"


class MultipleImageSerializer(serializers.Serializer):
    images = serializers.ListField(
        child=serializers.ImageField()
    )
