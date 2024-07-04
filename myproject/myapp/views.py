from django.shortcuts import render
from .neckmain import main as neck_main
from .back_main import main as back_main

def index(request):
    return render(request, 'myapp/index.html')

def analyze_neck(request):
    result = neck_main()
    return render(request, 'myapp/result.html', {'result': result, 'type': 'neck'})

def analyze_back(request):
    result = back_main()
    return render(request, 'myapp/result.html', {'result': result, 'type': 'back'})