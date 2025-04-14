import turtle

class GeoBeams:
    def __init__(self):
        self.screen = turtle.Screen()
        self.screen.setup(width=600, height=600)  # Adjust the screen size if necessary
        # Class attributes for beam coordinates and colors
        self.GEO_beam_coordinates = [
            (0,0), (190,0), (-190,0), (-190,190),
            (0,190), (190,190), (-190,-190), (0,-190), (190,-190)
        ]
        self.freq = [
            "navyblue", "lightgreen", "lime", "navyblue",
            "lightgreen", "lime", "lightgreen", 'lime', 'navyblue'
        ]
        # self.freq_sub is the frequency sub band allocation for each beam
        self.freq_sub = {
            "navyblue": [1,2,3],
            "lightgreen": [4,5],
            "lime": [6,7,8,9,10],
        }
        self.geo_beams = []  # List to hold turtle objects representing GEO beams
        self.create_and_position_beams()
        self.geo_user_coordinates =[
            (0,55), (30,0), (0,-70), (-80,0),
            (180, 60), (250, 10), (200, -60), (130, 10),
            (-180, 60), (-150, 10), (-180, -60), (-250, 10),
            (-180, 250), (-130, 200), (-200, 150), (-250, 200),
            (10, 250), (40, 200), (-10, 130), (-40, 200),
            (180, 250), (230, 180), (200, 130), (150, 180),
            (-180, -130), (-150, -180), (-180, -230), (-250, -180),
            (-10, -150), (60, -180), (-10, -250), (-60, -180),
            (180, -150), (250, -180), (180, -230), (130, -180)
        ]

        self.geo_users = []
        self.geo_user_beam = {}
        self.setup_geo_users()

    def create_and_position_beams(self):
        """Creates and positions turtle objects based on predefined coordinates and colors."""
        for coord, color in zip(self.GEO_beam_coordinates, self.freq):
            t = turtle.Turtle()
            t.shape("circle")
            t.shapesize(19/2)  # Approximation for GEO beam width 950km
            t.fillcolor(color)
            t.penup()  # Prevent drawing lines when moving
            t.speed("fastest")
            t.goto(coord)  # Move the turtle to the specified coordinates
            self.geo_beams.append(t)

    def setup_geo_users(self):
        for x, y in self.geo_user_coordinates:
            user_turtle = turtle.Turtle()
            user_turtle.penup()
            user_turtle.shape("square")
            user_turtle.fillcolor("GreenYellow")
            user_turtle.speed('fastest')
            user_turtle.goto(x, y)
            user_turtle.pendown()
            self.geo_users.append(user_turtle)

        # Assigning users to beams
        for user in self.geo_users:
            for beam in self.geo_beams:
                if user.distance(beam) <= 95:
                    self.geo_user_beam[user] = beam


