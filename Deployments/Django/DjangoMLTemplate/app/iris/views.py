from django.shortcuts import render
from django.http import JsonResponse
from . models import PredResults

import pandas as pd
import os

# Create your views here.

def predict(request):
    return render(request, 'iris/predict.html')


def predict_chances(request):

    # Receive data from client
    sepal_length = float(request.POST.get('sepal_length'))
    sepal_width = float(request.POST.get('sepal_width'))
    petal_length = float(request.POST.get('petal_length'))
    petal_width = float(request.POST.get('petal_width'))

    # Unpickle model
    model = pd.read_pickle("iris/new_model.pickle")
    # Make prediction
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    classification = result[0]

    PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                                petal_width=petal_width, classification=classification)

    print(f'classification is: {classification}')
    context = {
        'result': classification, 
        'sepal_length': sepal_length,
        'sepal_width': sepal_width, 
        'petal_length': petal_length, 
        'petal_width': petal_width}

    
    return render(request, 'iris/modal.html', context)
        


def results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "iris/results.html", data)