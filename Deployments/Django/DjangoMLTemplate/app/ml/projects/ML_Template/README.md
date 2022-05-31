# ML_Project_Template
Boilerplate for ML Projects

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

# create folds
python create_folds.py --folds 5

# train model
python train.py --fold 0 --model decision_tree_entropy