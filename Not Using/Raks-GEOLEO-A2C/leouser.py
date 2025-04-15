import turtle
import random

class LeoUserConfig:
    # Define coordinates as tuples (x, y)
    # LEO_A_USER_COORDINATES = [(-5, -5), (0, -75), (-75, 0), (-120, 150), (-150, 200), (-120, -150), (-150, -200), (75, -150), (150, -200)]
    # LEO_B_USER_COORDINATES = [(5, 5), (0, 75), (75, 0), (-250, 0), (-150, 50), (120, 50), (175, 75), (120, 150), (175, 175)]

    random.seed(42)
    LEO_A_USER_COORDINATES = [
        (-5, -5), (0, -75), (-75, 0), (-120, 150), (-150, 200),
        (-120, -150), (-150, -200), (75, -150), (150, -200), (-179, -5),
        (-103, -77), (145, 89), (4, -195), (-112, 148),(64, 169),
        (-144, -177), (-44, 33), (169, 137), (-45, 68),(-44, -17), (-127, -106)
    ]
    LEO_B_USER_COORDINATES = [
        (5, 5), (0, 75), (75, 0), (-250, 0), (-150, 50),
        (120, 50), (175, 75), (120, 150), (175, 175),
        (278, -187), (-154, -37), (-87, 24), (222, 277),
        (103, 65), (-192, -134), (-196, 254), (272, -16),
        (92, -39), (116, 76), (-220, 132), (-226, -188)
    ]

    def __init__(self, l1_all_turtles, l2_all_turtles, l3_all_turtles, l4_all_turtles):
        self.leo_user = []
        self.leo_user_B = []
        self.leo1_for_user = {}
        self.leo2_for_user = {}
        self.leo3_for_user = {}
        self.leo4_for_user = {}
        self.l1_all_turtles = l1_all_turtles
        self.l2_all_turtles = l2_all_turtles
        self.l3_all_turtles = l3_all_turtles
        self.l4_all_turtles = l4_all_turtles
        self.initialize_users()

    def initialize_users(self):
        # Initialize LEO A users
        for _ in self.LEO_A_USER_COORDINATES:
            leo_user_turtle = turtle.Turtle()
            self.leo_user.append(leo_user_turtle)

        for coordinates, user_turtle in zip(self.LEO_A_USER_COORDINATES, self.leo_user):
            self.configure_turtle(user_turtle, *coordinates)

        # Initialize LEO B users
        for _ in self.LEO_B_USER_COORDINATES:
            leo_user_turtle = turtle.Turtle()
            self.leo_user_B.append(leo_user_turtle)

        for coordinates, user_turtle in zip(self.LEO_B_USER_COORDINATES, self.leo_user_B):
            self.configure_turtle(user_turtle, *coordinates)

        self.assign_users()

    def configure_turtle(self, turtle, x, y):
        turtle.penup()
        turtle.shape("triangle")
        turtle.fillcolor("Cyan")
        turtle.speed('fastest')
        turtle.goto(x, y)
        turtle.pendown()


    def generate_random_list(self, list_len):
        # Initial list containing one of each number from 0 to 6
        random_list = list(range(7))

        # Add random numbers until the list has a total of 25 elements
        while len(random_list) < list_len:
            random_list.append(random.randint(0, 6))

        # Shuffle the list to randomize the order of elements
        random.shuffle(random_list)

        return random_list

    def assign_users(self):
        random_integers = self.generate_random_list(list_len=len(self.LEO_A_USER_COORDINATES)+1)
        # print(random_integers)

        for user_turtle, i in zip(self.leo_user, random_integers):
            self.leo1_for_user[user_turtle] = self.l1_all_turtles[i]
            self.leo3_for_user[user_turtle] = self.l3_all_turtles[i]

        for user_turtle, i in zip(self.leo_user_B, random_integers):
            self.leo2_for_user[user_turtle] = self.l2_all_turtles[i]
            self.leo4_for_user[user_turtle] = self.l4_all_turtles[i]


