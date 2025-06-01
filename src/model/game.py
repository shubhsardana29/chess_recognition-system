import tkinter as tk
from typing import Dict, Tuple
from model.chessboard_calibration import ChessboardCalibration
from model.board import Board
# from model.agent import Agent                                               REMOVED AGENT
from model.camera import Camera
from model.gui import GUI
from dotenv import dotenv_values
from pyqtgraph.Qt import QtCore, QtGui
from utils import draw_bounding_box_on_image

import cv2
import numpy as np
import time
import imutils
import PIL.Image as Image

# ADDED MORE LIBRARIES TO DISPLAY BOARD
import chess.svg
from IPython.display import SVG, display
import cairosvg
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output
from matplotlib import animation
from stockfish import Stockfish
stockfish = Stockfish(path="/opt/homebrew/bin/stockfish")
stockfish.set_skill_level(20)  # Set maximum skill level
stockfish.set_depth(20)  # Set search depth
stockfish.set_elo_rating(3000)  # Set high ELO rating

# ADDED
matplotlib.use('qt5agg')
# import PySimpleGUI as sg

# Global variables
virtualBoard = None
previousVirtualBoard = None
initialBoard = [[7, 3, 5, 9, 11, 5, 3, 7],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [6, 2, 4, 8, 10, 4, 2, 6]]

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

# sorteio de cores para não pegar sempre as mesmas
COLORS_INDEX = np.random.randint(0, len(STANDARD_COLORS), 12)
COLORS = [STANDARD_COLORS[i] for i in COLORS_INDEX]


class Game(GUI):
    __cam_address: str
    __running_calibration: ChessboardCalibration
    __board: Board
    __config: Dict
    # __agent: Agent                                                                            REMOVED AGENT
    __camera: Camera = None
    __fps: float
    __lastupdate: float
    __detections: list = None
    __scan_timer: QtCore.QTimer = None

    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        self.__config = dotenv_values()
        self.__cam_address = self.__config.get('CAM_ADDRESS')
        # self.__agent = Agent()                                                                    REMOVED AGENT
        self.__debug = bool(int(self.__config.get('DEBUG')))

        # frame rate metrics
        self.__fps = 0.
        self.__lastupdate = time.time()

    # ADDED PADDING AND ROTATE AS PASS IN PARAMETERS
    def mapping(self, padding, rotate):
        """
        Start mapping chess board
        """
        camera = Camera(self.__cam_address)
        frame = camera.capture()

        # do calibration mapping
        chessboard_calibration = ChessboardCalibration(debug=self.__debug)
        result = chessboard_calibration.mapping(                                                                 # CHANGE PARAMETER VALUES and ADDED NEW VARIABLE TO PREVENT EXCEPTION FROM ENDING EXECUTION
            chessboard_img=frame,
            fix_rotate=True,

            # rotate_val=90,
            # add_padding=True,

            # rotate_val=0,
            # add_padding=False,

            # rotate_val = 180

            # apply_kdilate=True,

            add_padding=padding,
            rotate_val=rotate,
            #   smooth_ksize=(17,17)
        )

        if result == None:
            print("Fail!")
            # RETURNS NONE IF RESULT IS NONE
            return 1

        result = chessboard_calibration.saveMapping()
        if result == 1:
            print("Fail!")
            # RETURNS NONE IF RESULT IS NONE
            return 1

        # release camera
        camera.destroy()
        print('Done!')

    def start(self):
        """
        Start game
        """
        self.__camera = Camera(self.__cam_address)

        self.__running_calibration = ChessboardCalibration()
        found, self.__board = self.__running_calibration.loadMapping()
        if not found:
            raise Exception('No mapping found. Run calibration mapping')

        initializeGame()  # MY FUNCTION

        # ADDED VARIABLES TO ALLOW ROTATION ADJUSTMENT
        global imageDisplayed
        imageDisplayed = False
        global rotationFixed
        rotationFixed = False

        self.__captureFrame()
        self.__runScan(only_prediction=True)
        self.show()

    def __captureFrame(self):
        global imageDisplayed
        global rotationFixed
        # ADDED ROTATION ADJUSTMENT
        if imageDisplayed and not rotationFixed:
            print("Adjust Rotation!")
            found, self.__board = self.__running_calibration.loadMapping(True)
            rotationFixed = True

            self.__processed_image = self.__running_calibration.applyMapping(
                self.__camera.capture())
            self.setImage(self.__addPiecesBoundingBoxes(
                self.__processed_image))

            self.__runScan()
        imageDisplayed = True

        frame = self.__camera.capture()
        self.__processed_image = self.__running_calibration.applyMapping(frame)

        result, hand_is_detected = self.__addHandBoundingBoxes(
            self.__processed_image)
        if hand_is_detected:
            self.__scheduleScan()
        # ADDED CONDITION TO NOT PRINT BOXES UNTIL ROTATION ADJUSTED
        elif rotationFixed:
            result = self.__addPiecesBoundingBoxes(self.__processed_image)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        self.setImage(result)
        self.__updateFrameRate()

        QtCore.QTimer.singleShot(1, self.__captureFrame)

    def __scheduleScan(self):
        self.__detections = None
        if self.__scan_timer is not None:
            self.__scan_timer.stop()

        self.__scan_timer = QtCore.QTimer(self)
        self.__scan_timer.timeout.connect(self.__runScan)
        self.__scan_timer.setSingleShot(True)
        self.__scan_timer.start(800)

    def __addPiecesBoundingBoxes(self, image):
        if self.__detections is None:
            return image

        image_pil = Image.fromarray(np.uint8(image.copy())).convert('RGB')
        height, width = image.shape[:2]

        for (name, bbox, acc, cls_id) in self.__detections:

            # Essas linhas que contem `np.random` faz com que os bounding boxes e a precisão
            # fiquem oscilando como se estivessem sendo detectados em real time.
            #
            # A ilusão faz parte do show :)
            #
            # - np.random.randint(-1, 1, len(bbox))                                                REMOVED RANDOM JITTER
            x, y, w, h = bbox
            # - np.random.uniform(0.01, 0)                                                                 REMOVED RANDOM ACCURACY FLUCTUATIONS
            acc = acc

            xmin = x / width
            ymin = y / height
            xmax = w / width
            ymax = h / height

            draw_bounding_box_on_image(
                image=image_pil,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                # color=COLORS[cls_id],
                # CHANGED PIECE COLOR TO BLUE
                color="Blue"
                # display_str_list=['{}: {:.1f}%'.format(name, acc * 100)]                                          REMOVED PIECE DETECTION LABEL
            )

        return np.array(image_pil)

    def __addHandBoundingBoxes(self, image):
        inverted = cv2.bitwise_not(image.copy())
        hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

        # ========
        # eliminar as casas
        # ========

        # ---- verdes
        lower = np.array([135, 6, 91])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv.copy(), lower, upper)
        mask = 255-mask
        green_squares_mask = mask.copy()

        # ---- brancas
        hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 81])
        upper = np.array([255, 43, 255])
        mask = cv2.inRange(hsv.copy(), lower, upper)
        mask = 255-mask
        white_squares_mask = mask.copy()

        # ---- resultado final sem as casas
        image_final = image.copy()
        image_final = cv2.bitwise_and(
            image_final, image_final, mask=green_squares_mask)
        image_final = cv2.bitwise_and(
            image_final, image_final, mask=white_squares_mask)

        # ========
        # Seleciona as peças
        # ========
        inverted = cv2.bitwise_not(image_final.copy())
        hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

        # ---- peças brancas
        target = image.copy()
        # Adjusted for better white piece detection
        lower = np.array([60, 50, 50])
        upper = np.array([255, 255, 255])
        white_pieces_mask = cv2.inRange(hsv.copy(), lower, upper)

        # ---- peças pretas
        # Adjusted for better black piece detection
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 100])
        black_pieces_mask = cv2.inRange(hsv.copy(), lower, upper)

        hand_is_detected, hand_contours = self.__hand_detected(
            image_final, white_pieces_mask, black_pieces_mask)
        if hand_is_detected:
            self.__drawHand(target, hand_contours)

        return (target, hand_is_detected)

    def __drawHand(self, target, hand_contours):
        peri = cv2.arcLength(hand_contours, True)
        biggest_cnt = cv2.approxPolyDP(hand_contours, 0.015 * peri, True)
        x, y, w, h = cv2.boundingRect(biggest_cnt)
        cv2.rectangle(target, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(target, 'HUMAN HAND', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def __hand_detected(self, no_houses_frame, white_pieces_mask, black_pieces_mask) -> Tuple[bool, list]:
        """
        return `True` or `False` if hand is detected
        """
        white_pieces_mask = 255-white_pieces_mask
        black_pieces_mask = 255-black_pieces_mask

        no_houses_frame = cv2.bitwise_and(
            no_houses_frame, no_houses_frame, mask=white_pieces_mask)
        no_houses_frame = cv2.bitwise_and(
            no_houses_frame, no_houses_frame, mask=black_pieces_mask)

        # convert image to gray scale
        gray = cv2.cvtColor(no_houses_frame, cv2.COLOR_BGR2GRAY)

        # This is the threshold level for every pixel.
        blur = cv2.GaussianBlur(gray, (11, 11), cv2.BORDER_DEFAULT)
        thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)[
            1]  # Lowered threshold from 70 to 50
        # Reduced erosions from 20 to 10
        thresh = cv2.erode(thresh, None, iterations=10)

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if cnts is not None and len(cnts) > 0:
            # Estou assumindo que a coisa maior na imagem diferente das casas e das peças
            # é uma mão, mas isso não é uma vdd absoluta.
            cnt = max(cnts, key=cv2.contourArea)
            return (True, cnt)
        else:
            return (False, None)

    def __updateFrameRate(self):
        now = time.time()
        dt = (now - self.__lastupdate)
        if dt <= 0:
            dt = 0.000000000001

        fps2 = 1.0 / dt
        self.__lastupdate = now
        self.__fps = self.__fps * 0.9 + fps2 * 0.1
        # self.print('Mean Frame Rate:  {:.2f} FPS'.format(self.__fps), index=0)                                      COMMENTED OUT LIVE FPS READING

    def __runScan(self, only_prediction: bool = False):
        global rotationFixed
        # Added condition to not print game board until rotation is adjusted
        if rotationFixed:
            print('scanning...')
            squares, self.__detections = self.__board.scan(
                self.__processed_image)
            board_state = self.__board.toMatrix(squares)

            # if not only_prediction:                                                                             COMMENTED OUT MOVE PRINTING + REMOVED AGENT
            #   human_move = self.__agent.state2Move(board_state)
            #   if human_move is not None:
            #     self.print('HUMAN: {}'.format(human_move.uci()))
            #     self.__agent.makeMove(human_move)
            #     self.__agent.updateState(board_state)

            # cpu_move = self.__agent.chooseMove()
            # if cpu_move is not None:
            #   self.print('BOT: {}'.format(cpu_move.uci()))
            #   self.__agent.makeMove(cpu_move)
            #   self.__agent.updateState(self.__agent.board.state())

            # print(board_state)
            # print(virtualBoard)
            print('SCAN:')
            printBoard(translate(board_state))
            # CALLING MY FUNCTION
            analyzeBoard(translate(board_state))

#################################################################################################################################################################################################################################

# EMPTY = -1
# W_PAWN = 0
# B_PAWN = 1
# W_KNIGHT = 2
# B_KNIGHT = 3
# W_BISHOP = 4
# B_BISHOP = 5
# W_ROOK = 6
# B_ROOK = 7
# W_QUEEN = 8
# B_QUEEN = 9
# W_KING = 10
# B_KING = 11


ROW = 0
COLUMN = 1

LOCATION = 0
DESTINATION = 1

WHITE = True
BLACK = False


def translate(board):
    translation = np.copy(board)

    for r in range(8):
        for c in range(8):
            if board[r][c] == 1:
                translation[r][c] = 6
            if board[r][c] == 2:
                translation[r][c] = 4
            if board[r][c] == 3:
                translation[r][c] = 2
            if board[r][c] == 4:
                translation[r][c] = 10
            if board[r][c] == 5:
                translation[r][c] = 8
            if board[r][c] == 6:
                translation[r][c] = 1
            if board[r][c] == 8:
                translation[r][c] = 5
            if board[r][c] == 9:
                translation[r][c] = 11
            if board[r][c] == 10:
                translation[r][c] = 9
            if board[r][c] == 11:
                translation[r][c] = 3

    return translation


def initializeGame():
    global previousScannedBoard, board, last_move, current_side, captureSquares, gameOver, lastMoveString, virtualBoard

    resetGame()

    previousScannedBoard = np.empty(0)
    board = chess.Board()
    last_move = None
    current_side = WHITE  # True = White, False = Black
    captureSquares = []
    gameOver = False
    lastMoveString = ''
    # Initialize virtualBoard with initial board state
    virtualBoard = np.copy(initialBoard)

    stockfish.set_position([])
    stockfish.set_skill_level(20)
    stockfish.set_depth(20)
    stockfish.set_elo_rating(3000)

    print("GAME STARTED")


def resetGame():
    global virtualBoard
    virtualBoard = np.copy(initialBoard)


def analyzeBoard(scannedBoard):
    global gameOver, previousScannedBoard, virtualBoard, initialBoard

    if gameOver:
        resetGame()

    newEmptySquares = []
    newOccupiedSquares = []
    changedColorSquares = []  # to determine a capture

    numberOfSquaresDifferentFromInitial = 0

    if len(previousScannedBoard) == 0:  # Only happens the first time this function is called
        previousScannedBoard = np.copy(scannedBoard)
        # Initialize virtualBoard with scanned board
        virtualBoard = np.copy(scannedBoard)
        print("SET PREVIOUS SCANNED BOARD")

    # First, update virtual board to match scanned board
    for r in range(8):
        for c in range(8):
            if scannedBoard[r][c] != virtualBoard[r][c]:
                if scannedBoard[r][c] == -1 and virtualBoard[r][c] > -1:
                    newEmptySquares.append([r, c])
                    print(
                        f"Empty square detected at {r},{c} (was {virtualBoard[r][c]})")
                elif scannedBoard[r][c] > -1 and virtualBoard[r][c] == -1:
                    newOccupiedSquares.append([r, c])
                    print(
                        f"Occupied square detected at {r},{c} (new piece {scannedBoard[r][c]})")
                elif scannedBoard[r][c] > -1 and virtualBoard[r][c] > -1:
                    changedColorSquares.append([r, c])
                    print(
                        f"Color change detected at {r},{c} (was {virtualBoard[r][c]}, now {scannedBoard[r][c]})")

                # Update virtual board to match scanned board
                virtualBoard[r][c] = scannedBoard[r][c]

            # Count differences from initial board
            if (scannedBoard[r][c] == -1 and initialBoard[r][c] > -1) or (scannedBoard[r][c] > -1 and initialBoard[r][c] == -1):
                numberOfSquaresDifferentFromInitial += 1

    print("\nBoard State Analysis:")
    print("New Empty Squares:", newEmptySquares)
    print("New Occupied Squares:", newOccupiedSquares)
    print("Changed Color Squares:", changedColorSquares)
    print("Virtual Board State:")
    printBoard(virtualBoard)
    print("Scanned Board State:")
    printBoard(scannedBoard)

    if numberOfSquaresDifferentFromInitial == 0:
        print("\nResetting game - board matches initial state")
        initializeGame()
        newEmptySquares = newOccupiedSquares = []
        gameOver = False
        runGameLogic(scannedBoard, newEmptySquares,
                     newOccupiedSquares, changedColorSquares)
    elif not gameOver:
        runGameLogic(scannedBoard, newEmptySquares,
                     newOccupiedSquares, changedColorSquares)


def runGameLogic(scannedBoard, newEmptySquares, newOccupiedSquares, changedColorSquares):
    global previousScannedBoard, best_move, current_side, captureSquares, gameOver, lastMoveString, last_move, virtualBoard

    if gameOver:
        print("Enter any key to restart game: ")
        input()
        initializeGame()

    illegal_move = []
    madeIllegalMove = False
    moveString = ''

    # If we have detected changes in the board state
    if len(newEmptySquares) > 0 or len(newOccupiedSquares) > 0 or len(changedColorSquares) > 0:
        print("Move detected!")

        if len(newEmptySquares) == 2 and not samePieceColor(newEmptySquares) and len(newOccupiedSquares) == 0:
            captureSquares = newEmptySquares
            print("Making Move!")
        elif len(newEmptySquares) + len(newOccupiedSquares) > 0:
            if len(newEmptySquares) + len(newOccupiedSquares) > 2 or len(newEmptySquares) > 1 or len(newOccupiedSquares) > 1 or len(newEmptySquares) == 0:
                print('ERROR: TOO MANY PIECES MOVED')
                illegal_move += newEmptySquares + newOccupiedSquares
            elif len(captureSquares) == 0 and len(newOccupiedSquares) == 0 and (len(changedColorSquares) == 0 or len(changedColorSquares) > 1 or (len(newEmptySquares) == 1 and (len(changedColorSquares) == 0 or len(changedColorSquares) > 1))):
                print('ERROR: INVALID MOVE DETECTED (possible capture)')
                illegal_move += newEmptySquares
            else:
                move = [newEmptySquares[0]]
                capturedPiece = False
                if len(newOccupiedSquares) == 1:
                    move.append(newOccupiedSquares[0])
                elif len(captureSquares) > 0:
                    if captureSquares[0] == move[0]:
                        move.append(captureSquares[1])
                    else:
                        move.append(captureSquares[0])
                    capturedPiece = True
                else:
                    move.append(changedColorSquares[0])
                    capturedPiece = True

                if isPromotion(move):
                    print(
                        "Input promotion piece:\nQueen -> q\nRook -> r\nBishop -> b\nKnight -> k")
                    piece = str(input())
                    if piece == 'r' or piece == 'b' or piece == 'k':
                        move.append(piece)
                    else:
                        move.append('q')

                if stockfish.is_move_correct(getMoveString(move)):
                    print('MOVED: ', end='')
                    updateBoard(move)
                    previousScannedBoard = np.copy(scannedBoard)
                    current_side = not current_side
                    illegal_move = []
                else:
                    print('ILLEGAL MOVE: ', end='')
                    illegal_move = move
                    madeIllegalMove = True

                moveString = getPrintingMoveString(move, capturedPiece)
                print(moveString)
                if illegal_move == []:
                    lastMoveString = moveString

        captureSquares = []
    else:
        print('NO MOVE')
        previousScannedBoard = np.copy(scannedBoard)
        captureSquares = []

    best_move = stockfish.get_best_move()
    if best_move == None:
        gameOver = True
    else:
        print("BEST MOVE: ", end='')
        print(best_move)

    squares = None
    if len(captureSquares) > 0:
        squares = dict.fromkeys(chess.SquareSet(
            getSquares(captureSquares)), "#ffff00")
    elif len(illegal_move) > 0:
        squares = dict.fromkeys(chess.SquareSet(
            getSquares(illegal_move)), "#cc0000cc")
    else:
        squares = {}  # Clear highlighting when no special cases

    if gameOver:
        bestMove = []
    else:
        bestMove = [(chess.parse_square(best_move[:2]),
                     chess.parse_square(best_move[2:4]))]

    # Only show last move if it was a legal move and not during capture
    if last_move and not len(captureSquares) > 0 and not len(illegal_move) > 0:
        last_move_to_show = last_move
        # Clear the last move after showing it
        last_move = None
    else:
        last_move_to_show = None

    # Create SVG board with current state
    if board.is_check():
        svg = chess.svg.board(board, lastmove=last_move_to_show, check=board.king(
            current_side), arrows=bestMove, size=500, fill=squares)
    else:
        svg = chess.svg.board(board, lastmove=last_move_to_show,
                              arrows=bestMove, size=500, fill=squares)

    img_png = cairosvg.svg2png(svg)
    img = Image.open(BytesIO(img_png))

    plt.ion()
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    if board.is_stalemate():
        plt.figtext(0.5, 0.95, "Stalemate!", ha="center", va="top", fontsize=18, bbox={
                    "facecolor": "orange", "alpha": 0.5, "pad": 5})
    elif board.is_checkmate():
        if current_side == BLACK:
            plt.figtext(0.5, 0.95, "Checkmate! - White Wins", ha="center", va="top",
                        fontsize=18, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
        else:
            plt.figtext(0.5, 0.95, "Checkmate! - Black Wins", ha="center", va="top",
                        fontsize=18, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    elif len(captureSquares) > 0:
        plt.figtext(0.5, 0.95, "Capturing...", ha="center", va="top", fontsize=18, bbox={
                    "facecolor": "orange", "alpha": 0.5, "pad": 5})
    elif madeIllegalMove:
        plt.figtext(0.5, 0.95, "Illegal move: " + moveString, ha="center", va="top",
                    fontsize=18, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    elif len(illegal_move) > 0:
        plt.figtext(0.5, 0.95, "Please fix board.", ha="center", va="top",
                    fontsize=18, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    if lastMoveString != '':
        plt.title('Last move: ' + lastMoveString, y=-.1)

    print('\n\n\n')


def samePieceColor(squares):
    return virtualBoard[squares[0][ROW]][squares[0][COLUMN]] % 2 == virtualBoard[squares[1][ROW]][squares[1][COLUMN]] % 2


def getSquares(setOfSquares):
    squares = []
    for square in setOfSquares:
        if len(square) == 2:
            squares.append(chess.parse_square(getCoordinate(square)))

    return squares


def isPromotion(move):
    # Checks that piece moving is a pawn
    if virtualBoard[move[LOCATION][ROW]][move[LOCATION][COLUMN]] == 0 or virtualBoard[move[LOCATION][ROW]][move[LOCATION][COLUMN]] == 1:
        # Checks that pawn is moving to bottom or top row
        if move[DESTINATION][ROW] == 0 or move[DESTINATION][ROW] == 7:
            # Checks that pawn is only moving one square up or down
            if abs(move[LOCATION][ROW] - move[DESTINATION][ROW]) == 1:
                print("Promotion!")
                return True

    return False


def getMoveString(move):
    if len(move) == 2:
        return getCoordinate(move[LOCATION]) + getCoordinate(move[DESTINATION])
    else:
        return getCoordinate(move[LOCATION]) + getCoordinate(move[DESTINATION]) + move[2]


def getPrintingMoveString(move, capturedPieceBool=False):
    string = ''
    string += getCoordinate(move[LOCATION])
    if capturedPieceBool:
        string += 'x'
    string += getCoordinate(move[DESTINATION])
    if board.is_checkmate():
        string += '#'
    elif board.is_check():
        string += '+'
    return string


def getCoordinate(square):
    coordinate = ''
    if square[COLUMN] == 0:
        coordinate += 'a'
    if square[COLUMN] == 1:
        coordinate += 'b'
    if square[COLUMN] == 2:
        coordinate += 'c'
    if square[COLUMN] == 3:
        coordinate += 'd'
    if square[COLUMN] == 4:
        coordinate += 'e'
    if square[COLUMN] == 5:
        coordinate += 'f'
    if square[COLUMN] == 6:
        coordinate += 'g'
    if square[COLUMN] == 7:
        coordinate += 'h'
    coordinate += str(8 - square[ROW])

    return coordinate


def updateBoard(move):
    global previousVirtualBoard, board, last_move, virtualBoard

    previousVirtualBoard = virtualBoard

    # Update the chess board
    move_string = getMoveString(move)
    last_move = board.push_san(move_string)

    # Update Stockfish position
    stockfish.make_moves_from_current_position([move_string])

    # Give Stockfish time to think about the next move
    stockfish.set_depth(20)
    stockfish.set_skill_level(20)

    # Update virtual board based on the chess board state
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        # Convert from chess square to array coordinates
        row = 7 - (square // 8)
        col = square % 8
        if piece.color == chess.WHITE:
            if piece.piece_type == chess.PAWN:
                virtualBoard[row][col] = 0
            elif piece.piece_type == chess.ROOK:
                virtualBoard[row][col] = 6
            elif piece.piece_type == chess.KNIGHT:
                virtualBoard[row][col] = 2
            elif piece.piece_type == chess.BISHOP:
                virtualBoard[row][col] = 4
            elif piece.piece_type == chess.QUEEN:
                virtualBoard[row][col] = 8
            elif piece.piece_type == chess.KING:
                virtualBoard[row][col] = 10
        else:  # BLACK pieces
            if piece.piece_type == chess.PAWN:
                virtualBoard[row][col] = 1
            elif piece.piece_type == chess.ROOK:
                virtualBoard[row][col] = 7
            elif piece.piece_type == chess.KNIGHT:
                virtualBoard[row][col] = 3
            elif piece.piece_type == chess.BISHOP:
                virtualBoard[row][col] = 5
            elif piece.piece_type == chess.QUEEN:
                virtualBoard[row][col] = 9
            elif piece.piece_type == chess.KING:
                virtualBoard[row][col] = 11

    # Clear any squares that don't have pieces
    for row in range(8):
        for col in range(8):
            if chess.square(col, 7-row) not in piece_map:
                virtualBoard[row][col] = -1

    print("\nUpdated Virtual Board:")
    printBoard(virtualBoard)
    print("\nChess Board State:")
    print(board)


# ADDED NEW FUNCTION
def printBoard(board, inverse=False):
    if inverse:
        pieces = ['•', '♙', '♟', '♘', '♞', '♗',
                  '♝', '♖', '♜', '♕', '♛', '♔', '♚']
    else:
        pieces = ['•', '♟', '♙', '♞', '♘', '♝',
                  '♗', '♜', '♖', '♛', '♕', '♚', '♔']

    row = 8
    for r in range(8):
        print(row, end=" ")
        row -= 1
        for c in range(8):
            print(pieces[board[r][c] + 1], end=' ')
        print()
    print("  a b c d e f g h ")


def getPromotedPiece(piece):
    pieceNumber = None
    if piece == 'q':
        pieceNumber = 8
    if piece == 'r':
        pieceNumber = 6
    if piece == 'b':
        pieceNumber = 4
    if piece == 'k':
        pieceNumber = 2

    if current_side == BLACK:
        pieceNumber += 1

    return pieceNumber
