"""
Coffee Factory: A multiple producer - multiple consumer approach

Generate a base class Coffee which knows only the coffee name
Create the Espresso, Americano and Cappuccino classes which inherit the base class knowing that
each coffee type has a predetermined size.
Each of these classes have a get message method

Create 3 additional classes as following:
    * Distributor - A shared space where the producers puts coffees and the consumers takes them
    * CoffeeFactory - An infinite loop, which always sends coffees to the distributor
    * User - Another infinite loop, which always takes coffees from the distributor

The scope of this exercise is to correctly use threads, classes and synchronization objects.
The size of the coffee (ex. small, medium, large) is chosen randomly everytime.
The coffee type is chosen randomly everytime.

Example of output:

Consumer 65 consumed espresso
Factory 7 produced a nice small espresso
Consumer 87 consumed cappuccino
Factory 9 produced an italian medium cappuccino
Consumer 90 consumed americano
Consumer 84 consumed espresso
Factory 8 produced a strong medium americano
Consumer 135 consumed cappuccino
Consumer 94 consumed americano
"""
import sys
from threading import Thread, Semaphore
import random

class Coffee: # pylint: disable=E1101
    """ Base class """
    def __init__(self):
        pass

    def get_name(self):
        """ Returns the coffee name """
        return self.name

    def get_size(self):
        """ Returns the coffee size """
        return self.size

class Espresso(Coffee):
    """ Espresso implementation """
    def __init__(self, size):
        self.name = "espresso"
        self.size = size

    def get_message(self):
        """ Output message """
        return f"Heavy {self.size} espresso"

class Americano(Coffee):
    """ Americano implementation """
    def __init__(self, size):
        self.name = "americano"
        self.size = size

    def get_message(self):
        """ Output message """
        return f"Strong {self.size} americano"

class Cappuccino(Coffee):
    """ Cappuccino implementation """
    def __init__(self, size):
        self.name = "cappuccino"
        self.size = size

    def get_message(self):
        """ Output message """
        return f"Italian {self.size} cappuccino"


CoffeeType = [Espresso, Americano, Cappuccino]
Size = ["small", "medium", "large"]


class Distributor:
    """ Distributor implementation """
    def __init__(self, size):
        self.ready_coffees = []

        self.sem_empty = Semaphore(value = size)
        self.sem_full = Semaphore(value = 0)

    def put(self, coffee):
        """ Puts a coffee """
        self.ready_coffees.append(coffee)

    def get(self):
        """ Gets a coffee """
        return self.ready_coffees.pop()

class CoffeeFactory: # pylint: disable=too-few-public-methods
    """ CoffeeFactory implementation """
    def __init__(self, dist, no_producer, total_to_produce):
        self.dist = dist
        self.no_producer = no_producer
        self.total_to_produce = total_to_produce

    def run(self):
        """ Produces coffee """
        while self.total_to_produce > 0:
            self.dist.sem_empty.acquire()
            self.total_to_produce -= 1
            coffe_type = random.choice(CoffeeType)
            coffee_size = random.choice(Size)
            self.dist.put(coffe_type(coffee_size))
            print(f"Factory {self.no_producer} produced a nice {coffee_size} {coffe_type.__name__}")
            self.dist.sem_full.release()

class User: # pylint: disable=too-few-public-methods
    """ User implementation """
    def __init__(self, dist, no_producer, total_to_consume):
        self.dist = dist
        self.no_producer = no_producer
        self.total_to_consume = total_to_consume

    def run(self):
        """ Consumes coffee """
        while self.total_to_consume > 0:
            self.dist.sem_full.acquire()
            self.total_to_consume -= 1
            coffee = self.dist.get()
            print(f"Consumer {self.no_producer} consumed a nice "
                  f"{coffee.get_size()} {coffee.get_name()}")
            self.dist.sem_empty.release()

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python task2.py <total_producers> <total_consumers> "
              "<size_of_buffer> <total_coffees_to_produce_per_producer> "
              "<total_coffees_to_consume_per_consumer>")
        sys.exit(1)
    total_producers = int(sys.argv[1])
    total_consumers = int(sys.argv[2])
    size_of_buffer = int(sys.argv[3])
    coffees_per_producer = int(sys.argv[4])
    coffees_per_consumer = int(sys.argv[5])

    print(f"Total producers: {total_producers}"
        f"\nTotal consumers: {total_consumers}"
        f"\nSize of buffer: {size_of_buffer}"
        f"\nTotal coffees to produce per producer: {coffees_per_producer}"
        f"\nTotal coffees to consume per consumer: {coffees_per_consumer}"
        f"\n\n")

    if (total_producers * coffees_per_producer) < (total_consumers * coffees_per_consumer):
        print("Total number of coffees produced must be greater than (or at "
              "least equal to) the total number of coffees consumed")
        sys.exit(1)
    if (total_producers * coffees_per_producer) - (total_consumers * coffees_per_consumer) \
        > size_of_buffer:
        print("The size of the buffer must be greater than (or at least equal "
              "to) the difference between the total number of coffees produced "
              "and the total number of coffees consumed. Otherwise, the buffer "
              "will overflow and the program will crash.")
        sys.exit(1)

    distributor = Distributor(size_of_buffer)

    threads = []
    for i in range(total_producers):
        threads.append(Thread(target=CoffeeFactory(distributor, i, coffees_per_producer).run))

    for i in range(total_consumers):
        threads.append(Thread(target=User(distributor, i, coffees_per_consumer).run))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("\n\nDone!")
    print(f"Total coffees left: {len(distributor.ready_coffees)}")
