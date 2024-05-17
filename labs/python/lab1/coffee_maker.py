"""
A command-line controlled coffee maker.

Implement the coffee maker's commands. Interact with the user via stdin and print to stdout.

Requirements:
    - use functions
    - use __main__ code block
    - access and modify dicts and/or lists
    - use at least once some string formatting (e.g. functions such as strip(), lower(),
    format()) and types of printing (e.g. "%s %s" % tuple(["a", "b"]) prints "a b"
    - BONUS: read the coffee recipes from a file, put the file-handling code in another module
    and import it (see the recipes/ folder)

There's a section in the lab with syntax and examples for each requirement.

Feel free to define more commands, other coffee types, more resources if you'd like and have time.

Tips:
*  Start by showing a message to the user to enter a command, remove our initial messages
*  Keep types of available coffees in a data structure such as a list or dict
e.g. a dict with coffee name as a key and another dict with resource mappings (resource:percent)
as value
"""

import sys
import load_recipes

# Commands
EXIT = "exit"
LIST_COFFEES = "list"
MAKE_COFFEE = "make"  #!!! when making coffee you must first check that you have enough resources!
HELP = "help"
REFILL = "refill"
RESOURCE_STATUS = "status"
commands = [EXIT, LIST_COFFEES, MAKE_COFFEE, REFILL, RESOURCE_STATUS, HELP]
commands_help = {
    EXIT: "exit the coffee maker",
    LIST_COFFEES: "list the available coffee types",
    MAKE_COFFEE: "make a coffee: you must first check the available recipes",
    REFILL: "refill the resources",
    RESOURCE_STATUS: "show the status of the resources",
    HELP: "show the available commands: this message"
}

# Coffees and resources
coffees = []
resources = []
RESOURCES = {}

"""
Example result/interactions:

I'm a smart coffee maker
Enter command:
list
americano, cappuccino, espresso
Enter command:
status
water: 100%
coffee: 100%
milk: 100%
Enter command:
make
Which coffee?
espresso
Here's your espresso!
Enter command:
refill
Which resource? Type 'all' for refilling everything
water
water: 100%
coffee: 90%
milk: 100%
Enter command:
exit
"""


def list_coffees():
    """Function listing the available coffee types."""
    load_recipes.list_coffee_types()

def make_coffee():
    """Function for making a coffee."""
    print("Which coffee?")
    coffee = input()

    # Make coffee
    recipe = load_recipes.load_recipes(coffee, RESOURCES)
    if not recipe:
        return
    # Deduct resources
    for resource, percent in recipe:
        RESOURCES[resource] -= int(percent)

    print(f"Here's your {coffee}!")

def refill():
    """Function for refilling the resources."""
    print("Which resource? Type 'all' for refilling everything")
    resource = input()
    if resource == "all":
        RESOURCES.update({resource: 100 for resource in resources})
    else:
        RESOURCES[resource] = 100

def resource_status():
    """Function for showing the status of the resources."""
    print("\n".join(f"{resource}: {RESOURCES[resource]}%" for resource in resources))

def help_commands():
    """Function for showing the available commands."""
    print("Available commands:")
    print("\n".join(f"{command}: {commands_help[command]}" for command in commands))

def switch(command_type):
    """Function for switching between the commands."""
    if command_type == EXIT:
        sys.exit()
    elif command_type == LIST_COFFEES:
        list_coffees()
    elif command_type == MAKE_COFFEE:
        make_coffee()
    elif command_type == REFILL:
        refill()
    elif command_type == RESOURCE_STATUS:
        resource_status()
    elif command_type == HELP:
        help_commands()
    else:
        print("Invalid command. Type 'help' for a list of commands")

if __name__ == "__main__":
    print("I'm a simple coffee maker")
    coffees, resources, RESOURCES = load_recipes.setup_recipes()
    while True:
        print("Enter command:")
        command = input()
        switch(command)
