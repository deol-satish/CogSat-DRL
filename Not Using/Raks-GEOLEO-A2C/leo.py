import turtle
import random

class BaseTurtle:
    def __init__(self, x, y, color, direction=90):
        self.turtle_obj = turtle.Turtle()
        self.turtle_obj.penup()
        self.turtle_obj.speed('fastest')
        self.draw_circle(x, y, color)
        self.turtle_obj.setheading(direction)
        self.turtle_obj.pendown()

    def draw_circle(self, x, y, fill_color):
        self.turtle_obj.goto(x, y)
        self.turtle_obj.shape("circle")
        self.turtle_obj.fillcolor(fill_color)
        self.turtle_obj.shapesize(6)  # since LEO beam width is 600km

class LEOTurtle(BaseTurtle):
    leo_colors = ["orange", "magenta", "green", "blue", "purple", "pink"]

    def __init__(self, x, y, central_color="red", direction=90):
        super().__init__(x, y, central_color, direction)
        self.initial_x = x
        self.initial_y = y
        self.all_turtles = [self.turtle_obj]
        self.chosen_colors = set(central_color)
        # self.freq_sub is the frequency sub band allocation for each beam, associated with color for ease of use
        self.freq_sub = {
            "red": [1],
            "orange": [2],
            "magenta": [3],
            "green": [4],
            "blue": [5],
            "purple": [6],
            "pink": [7],
            'aqua': [8],
            'khaki': [9],
            'mistyrose': [10]
        }

        for _ in range(6):
            color = self.choose_random_color()
            if color:
                new_turtle = BaseTurtle(x, y, color)
                self.all_turtles.append(new_turtle.turtle_obj)

        self.position_turtles(direction)

    def choose_random_color(self):
        available_colors = list(set(self.leo_colors) - self.chosen_colors)
        if not available_colors:
            return None
        chosen_color = random.choice(available_colors)
        self.chosen_colors.add(chosen_color)
        return chosen_color

    def position_turtles(self, direction):
        for idx, turtle_obj in enumerate(self.all_turtles[1:], start=1):
            turtle_obj.penup()
            turtle_obj.speed('fastest')
            turtle_obj.setheading(60 * idx)
            turtle_obj.forward(120)
            turtle_obj.setheading(direction)
            turtle_obj.pendown()

    def move(self, distance, angle):
        """Move all turtles in the group diagonally."""
        for turtle_obj in self.all_turtles:
            turtle_obj.setheading(angle)
            turtle_obj.penup()
            turtle_obj.forward(distance)
            turtle_obj.pendown()

    def get_coordinates(self):
        coordinates = []
        for turtle_obj in self.all_turtles:
            x, y = turtle_obj.position()
            coordinates.append((x, y))  # Append the position as a tuple
        return coordinates

    def reset_positions(self, direction):
        for turtle_obj in self.all_turtles:
            turtle_obj.penup()
            turtle_obj.goto(self.initial_x, self.initial_y)
            turtle_obj.pendown()

        for idx, turtle_obj in enumerate(self.all_turtles[1:], start=1):
            turtle_obj.penup()
            turtle_obj.speed('fastest')
            turtle_obj.setheading(60 * idx)
            turtle_obj.forward(120)
            turtle_obj.setheading(direction)
            turtle_obj.pendown()