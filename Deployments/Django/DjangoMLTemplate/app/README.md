# DJANGO ML Project Template
Boilerplate for ML Projects


# Steps:

* Install Django project dependencies from requirements.txt
* Install ML dependencies from /ml/projects/ML_Template/requirements.txt

1. Create new folder named 'input' in project directory
2. Place data files for model in input folder
3. Update src/config.py with paths, models, other params
4. Run ml mangement command: python manage.py load_projects update models with new project

- Perform exploration using notebooks in notebook direcotory
- Trained models are saved to models directory by default
- Use run.sh to take advantage of argparse 
- api endpoints will autopopulate via 

# Commands:

# move to src directory
cd /src/

# File Structure

```bash
< PROJECT ROOT >
├───charts
│   ├───data
│   ├───management
│   │   └───commands
│   │       └───__pycache__
│   ├───migrations
│   │   └───__pycache__
│   └───__pycache__
├───DjangoApp
│   └───__pycache__
├───iris
│   ├───migrations
│   └───__pycache__
├───ml
│   ├───management
│   │   └───commands
│   │       └───__pycache__
│   ├───migrations
│   ├───projects
│   │   ├───ML_Template
│   │   │   ├───input
│   │   │   ├───metrics
│   │   │   ├───models
│   │   │   ├───notebooks
│   │   │   │   └───.ipynb_checkpoints
│   │   │   └───src
│   │   │       └───__pycache__
│   │   └───Store_Timeseries
│   │       ├───input
│   │       ├───metrics
│   │       ├───models
│   │       ├───notebooks
│   │       │   └───.ipynb_checkpoints
│   │       └───src
│   │           └───__pycache__
│   └───__pycache__
├───stocks
│   ├───migrations
│   ├───src
│   │   └───__pycache__
│   └───__pycache__
└───templates
    ├───charts
    │   └───partials
    ├───iris
    ├───ml
    ├───partials
    └───stocks
```
=======

# Steps:

1. Create new folder named 'input' in project directory
2. Place data files for model in input folder
3. Update src/config.py with paths, models, other params
4. Run ml mangement command: python manage.py load_projects update models with new project

- Perform exploration using notebooks in notebook direcotory
- Trained models are saved to models directory by default
- Use run.sh to take advantage of argparse 
- api endpoints will autopopulate via 

# Commands:

# move to src directory
cd /src/
