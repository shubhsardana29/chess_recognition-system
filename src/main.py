import sys
from argparse import ArgumentParser, BooleanOptionalAction
from model import Game
from PyQt5.QtWidgets import QApplication  # Updated import
import time  # Added library

# Define arguments
parser = ArgumentParser()
parser.add_argument("-m", "--mapping",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Starts the mapping of the board")

parser.add_argument("-s", "--start",
                    action=BooleanOptionalAction,
                    default=False,
                    help="Chess game starts")

args = vars(parser.parse_args())

if __name__ == "__main__":
    # Calibration mapping
    if args['mapping']:
        app = QApplication(sys.argv)  # Updated to use QApplication from PyQt5
        game = Game()
        result = game.mapping(padding=False, rotate=0)  # Start mapping with default values
        while result == 1:
            result = game.mapping(padding=True, rotate=0)  # Continue mapping if result is 1
        sys.exit(app.exec_())  # End the application event loop after mapping

    # Start a game
    if args['start']:
        app = QApplication(sys.argv)  # Updated to use QApplication from PyQt5
        game = Game()

        # Prompt user for padding
        print("Padding? y or n: ", end='')  # Added user prompt for padding
        padding = None
        value = input().strip().lower()  # Added strip() and lower() to avoid case issues
        if value == 'y': 
            padding = True
        else: 
            padding = False

        # Uncomment if you need board orientation input
        print("Input board orientation (as viewed from camera): 1 (White on bottom), 2 (White on left), 3 (White on top), 4 (White on right): ")
        rotate = None
        value = input()
        if int(value) == 1: rotate = 270
        elif int(value) == 2: rotate = 180
        elif int(value) == 3: rotate = 90
        else: rotate = 0

        # Mapping and game start
        result = game.mapping(padding, 0)  # Start mapping with padding value and default rotation
        while result == 1: 
            result = game.mapping(padding, 0)  # Continue mapping if result is 1

        game.start()  # Start the game

        sys.exit(app.exec_())  # Start the application event loop


# import sys
# from argparse import ArgumentParser, BooleanOptionalAction
# from model import Game
# from PyQt5.QtWidgets import QApplication

# def main():
#     parser = ArgumentParser()
#     parser.add_argument("-m", "--mapping",
#                         action=BooleanOptionalAction,
#                         default=False,
#                         help="Starts the mapping of the board")
#     parser.add_argument("-s", "--start",
#                         action=BooleanOptionalAction,
#                         default=False,
#                         help="Chess game starts")
#     args = parser.parse_args()

#     if args.mapping:
#         app = QApplication(sys.argv)
#         game = Game()
#         result = game.mapping(padding=False, rotate=0)
#         while result == 1:
#             result = game.mapping(padding=True, rotate=0)
#         sys.exit(app.exec_())

#     elif args.start:
#         app = QApplication(sys.argv)
#         game = Game()

#         # ask for padding
#         val = input("Padding? (y/n): ").strip().lower()
#         padding = (val == 'y')

#         # ask for orientation
#         orient = input(
#           "Board orientation (1=White bottom, 2=left, 3=top, 4=right): "
#         ).strip()
#         rotate = {'1':270, '2':180, '3':90}.get(orient, 0)

#         # do the mapping pass
#         result = game.mapping(padding=padding, rotate=rotate)
#         while result == 1:
#             result = game.mapping(padding=padding, rotate=rotate)

#         # launch the game
#         game.start()
#         sys.exit(app.exec_())

# if __name__ == "__main__":
#     main()
