from django.core.management.base import BaseCommand
from django.conf import settings
from ml.models import Project
import os

class Command(BaseCommand):
    help = 'Load *projects *models *notebooks *configs from projects diectory into ml.models'

    def handle(self, *args, **kwargs):

        path = settings.BASE_DIR / 'ml' / 'projects'
        projects = [x for x in os.listdir(path) if x != '__init__.py']
        print(projects) 
        #  >>> ['ML_Template', 'Store_Timeseries'] 

        Project.objects.bulk_create(projects)



        # dir_projects = 0 
        # if dir_projects == Project.objects.count(): 
        #     pass
        

        # Use import os to load directory/file info to variables
        # Use import pathlib to load directory/file info to variables

        # Module.objects.all().delete()
        # course_names = [
        #     'Computer Science', 'Mathematics', 'Physics', 'Film Studies'
        # ]

        # if not Course.objects.count():
        #     for course_name in course_names:
        #         Course.objects.create(name=course_name)

        # # Computer Science
        # cs = Course.objects.get(name='Computer Science')

        # compsci_modules = [
        #     'AI',
        #     'Machine Learning',
        #     'Web Development',
        #     'Software Engineering',
        #     'NoSQL Databases'
        # ]

        # for module in compsci_modules:
        #     Module.objects.create(name=module, course=cs)
