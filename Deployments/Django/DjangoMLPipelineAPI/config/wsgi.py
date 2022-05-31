"""
WSGI config for config project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_wsgi_application()


# ML Registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.random_forest import RandomForestClassifier
from apps.ml.income_classifier.extra_trees import ExtraTreesClassifier

try:
    registry = MLRegistry() # create ML Registry
    # Random Forest Classifier
    rf = RandomForestClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name='income_classifier', 
                            algorithm_object=rf,
                            algorithm_name='random forest',
                            algorithm_status='production',
                            algorithm_version='0.0.1',
                            owner='Mike',
                            algorithm_description="Random Forest with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(RandomForestClassifier))

    # Extra Trees Classifier
    et = ExtraTreesClassifier
    # add to ML registry
    registry.add_algorithm(endpoint_name ='income_classifier',
                            algorithm_object = et,
                            algorithm_name = 'extra trees',
                            algorithm_status = 'production',
                            algorithm_version='0.0.1',
                            owner='Mike',
                            algorithm_description = 'Extra Trees with simple pre- and post-processing',
                            algorithm_code = inspect.getsource(RandomForestClassifier))

except Exception as e:
    print('Exception while loading the algorithms to the registry,', str(e))
