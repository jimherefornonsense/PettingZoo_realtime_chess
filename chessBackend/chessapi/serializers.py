from rest_framework import serializers
from .models import BoardInstance


class MyBoardSerializer(serializers.ModelSerializer):
    class Meta:
        model = BoardInstance
        fields = ('uuid', 'time', 'board')
