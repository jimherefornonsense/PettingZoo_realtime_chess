from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import MyBoardSerializer
from .models import BoardInstance
from django.db.models import Q


@api_view(['POST'])
def create_my_model(request):
    serializer = MyBoardSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    else:
        return Response(serializer.errors, status=400)


@api_view(['GET'])
def get_my_models_grouped_by_uuid(request):
    uuids = BoardInstance.objects.values_list('uuid', flat=True).distinct()
    grouped_models = []
    for uuid in uuids:
        my_models = BoardInstance.objects.filter(Q(uuid=uuid))
        serializer = MyBoardSerializer(my_models, many=True)
        grouped_models.append(serializer.data)
    return Response(grouped_models)
