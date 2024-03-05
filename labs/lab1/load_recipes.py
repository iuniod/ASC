"""
    Bonus task: load all the available coffee recipes from the folder 'recipes/'
    File format:
        first line: coffee name
        next lines: resource=percentage

    info and examples for handling files:
        http://cs.curs.pub.ro/wiki/asc/asc:lab1:index#operatii_cu_fisiere
        https://docs.python.org/3/library/io.html
        https://docs.python.org/3/library/os.path.html
"""

import os
RECIPES_FOLDER = "recipes"
coffee_types = []
resources_types = []
recipes = []

def setup_recipes():
    """Function setting up the recipes."""
    # Read the recipes from the file and create a list of coffee types
    dir_list = os.listdir("recipes/")

    # Read the recipe from the file
    for file in dir_list:
        with open(os.path.join(RECIPES_FOLDER, file), "r", encoding="utf-8") as file:
            recipe = file.read().split("\n")
            recipe = [line.split("=") for line in recipe if line]

        # Add the coffee type and resources to the list
        coffee_types.append(recipe[0][0])
        for resource, _ in recipe[1:]:
            if resource not in resources_types:
                resources_types.append(resource)

        # Add the recipe to the list
        recipes.append(recipe[1:])

    return coffee_types, resources_types, {resource: 100 for resource in resources_types}

def load_recipes(coffee_type, resources):
    """ Function making a coffee."""
    # Check if the coffee_type is in the resources
    if coffee_type not in coffee_types:
        print("Coffee type not found")
        return None

    # Read the recipe for the coffee_type
    recipe = recipes[coffee_types.index(coffee_type)][1:]

    # Check if the resources are enough for the recipe
    for resource, percent in recipe:
        if resources[resource] < int(percent):
            print("Not enough resources")
            return None

    return recipe

def list_coffee_types():
    """ Function listing the available coffee types."""
    # Print the available coffee types
    print(", ".join(coffee_types))
