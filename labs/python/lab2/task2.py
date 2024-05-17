"""
    Basic thread handling exercise:

    Use the Thread class to create and run more than 10 threads which print their name and a random
    number they receive as argument. The number of threads must be received from the command line.

    e.g. Hello, I'm Thread-96 and I received the number 42

"""
import sys
import random
from threading import Thread

def handler(thread_no, number):
    """ Thread handler """
    print(f"Hello, I'm Thread-{thread_no} and I received the number {number}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python task2.py <number_of_threads>")
        sys.exit(1)

    number_of_threads = int(sys.argv[1])
    if number_of_threads < 10:
        print("Number of threads must be at least 10")
        sys.exit(1)

    threads = []

    for i in range(number_of_threads):
        threads.append(Thread(target=handler, args=(i, random.randint(1, 100))))

    for t in threads:
        t.start()

    for t in threads:
        t.join()
