import pygame
import sys
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model, Model
from pygame.locals import *

WINDOWSIZEX = 1200
WINDOWSIZEY = 680
BOUNDRYINC = 10

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Load CNN-SVM
cnn_model = load_model("cnn_svm_feature_extractor.keras")
svm_model = joblib.load("cnn_svm_classifier.pkl")

feature_extractor = Model(
    inputs=cnn_model.input,
    outputs=cnn_model.get_layer("feature_layer").output
)

LABELS = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9"}

pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 40)
SMALLFONT = pygame.font.Font("freesansbold.ttf", 22)

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Recognition CNN-SVM")
DISPLAYSURF.fill(WHITE)

iswriting = False
number_xcord = []
number_ycord = []
last_processed = None


def preprocess_for_mnist(gray_img):
    # Invert so digit becomes white on black
    gray = 255 - gray_img

    # Threshold to remove weak noise
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find non-zero region
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]

    # Add padding around digit
    digit = np.pad(digit, ((20, 20), (20, 20)), mode='constant', constant_values=0)

    # Resize to 28x28
    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    digit = digit.astype("float32") / 255.0

    return digit


def predict_digit(processed_img):
    img_input = processed_img.reshape(1, 28, 28, 1)
    features = feature_extractor.predict(img_input, verbose=0)
    pred = svm_model.predict(features)[0]
    return pred


while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, BLUE, (xcord, ycord), 10, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False

            if len(number_xcord) == 0 or len(number_ycord) == 0:
                continue

            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x = max(number_xcord[0] - BOUNDRYINC, 0)
            rect_max_x = min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_y = max(number_ycord[0] - BOUNDRYINC, 0)
            rect_max_y = min(WINDOWSIZEY, number_ycord[-1] + BOUNDRYINC)

            number_xcord = []
            number_ycord = []

            # Capture only drawn region
            img_arr = pygame.surfarray.array3d(DISPLAYSURF)
            img_arr = img_arr[rect_min_x:rect_max_x, rect_min_y:rect_max_y]

            if img_arr.size == 0:
                continue

            # Convert to OpenCV shape
            img_arr = np.transpose(img_arr, (1, 0, 2))

            # Convert to grayscale
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

            processed = preprocess_for_mnist(gray)

            if processed is None:
                continue

            last_processed = processed.copy()

            pred = predict_digit(processed)
            label = LABELS[pred]

            # Clear text area
            pygame.draw.rect(DISPLAYSURF, WHITE, (20, 20, 420, 90))

            # Show prediction
            textSurface = FONT.render("Prediction: " + label, True, BLACK)
            DISPLAYSURF.blit(textSurface, (20, 20))

            hintSurface = SMALLFONT.render("Press N to clear", True, BLACK)
            DISPLAYSURF.blit(hintSurface, (20, 65))

        if event.type == KEYDOWN:
            if event.unicode.lower() == "n":
                DISPLAYSURF.fill(WHITE)
                last_processed = None

    # Show processed 28x28 preview
    if last_processed is not None:
        preview = (last_processed * 255).astype(np.uint8)
        preview = cv2.resize(preview, (140, 140), interpolation=cv2.INTER_NEAREST)
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
        preview_surface = pygame.surfarray.make_surface(np.transpose(preview, (1, 0, 2)))

        pygame.draw.rect(DISPLAYSURF, WHITE, (950, 20, 180, 180))
        DISPLAYSURF.blit(preview_surface, (970, 40))

    pygame.display.update()