import cv2
import numpy as np
from skimage import measure
import imutils


def sort_contours(character_contours):
    """Sort contours left to right by x position."""
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(
        *sorted(zip(character_contours, boundingBoxes),
                key=lambda b: b[1][0],  # sort by x (left to right)
                reverse=False)
    )
    return character_contours


def segment_chars(plate_img, fixed_width=400):
    """
    Extract characters from a license plate image.
    Uses HSV Value channel + adaptive thresholding + connected components.
    Returns list of character image crops, or None.
    """
    # Extract Value channel from HSV
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    # Adaptive threshold to reveal characters
    thresh = cv2.adaptiveThreshold(
        V, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    thresh = cv2.bitwise_not(thresh)

    # Resize to canonical width
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Connected components analysis
    labels = measure.label(thresh, background=0)
    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255

        cnts, _ = cv2.findContours(
            labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = 0.5 < heightRatio < 0.95

            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    contours, _ = cv2.findContours(
        charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    characters = []
    if contours:
        contours = sort_contours(contours)
        addPixel = 4

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            y = max(0, y - addPixel)
            x = max(0, x - addPixel)
            temp = bgr_thresh[
                y: y + h + (addPixel * 2),
                x: x + w + (addPixel * 2)
            ]
            characters.append(temp)

        return characters
    else:
        return None


class PlateFinder:
    """
    Detects license plates in an image using edge detection,
    morphological operations, and contour filtering.
    """

    def __init__(self, minPlateArea=4100, maxPlateArea=15000):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, ksize=(22, 3)
        )
        # Public attributes populated after find_possible_plates()
        self.char_on_plate = []
        self.corresponding_area = []
        self.after_preprocess = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, input_img):
        """Blur → grayscale → Sobel X → Otsu threshold → morphological close."""
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, threshold_img = cv2.threshold(
            sobelx, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        morph = threshold_img.copy()
        cv2.morphologyEx(
            src=threshold_img,
            op=cv2.MORPH_CLOSE,
            kernel=self.element_structure,
            dst=morph
        )
        return morph

    def extract_contours(self, preprocessed):
        contours, _ = cv2.findContours(
            preprocessed,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_NONE
        )
        return contours

    # ------------------------------------------------------------------
    # Plate validation helpers
    # ------------------------------------------------------------------

    def ratioCheck(self, area, width, height):
        """Strict ratio check used after cleaning."""
        ratio = float(width) / float(height) if height != 0 else 0
        if ratio < 1:
            ratio = 1 / ratio
        if (area < self.min_area or area > self.max_area) or \
           (ratio < 3 or ratio > 6):
            return False
        return True

    def preRatioCheck(self, area, width, height):
        """Loose ratio check used before cleaning."""
        ratio = float(width) / float(height) if height != 0 else 0
        if ratio < 1:
            ratio = 1 / ratio
        if (area < self.min_area or area > self.max_area) or \
           (ratio < 2.5 or ratio > 7):
            return False
        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect
        if width > height:
            angle = -rect_angle
        else:
            angle = 90 + rect_angle
        if angle > 15:
            return False
        if height == 0 or width == 0:
            return False
        area = width * height
        return self.preRatioCheck(area, width, height)

    # ------------------------------------------------------------------
    # Plate cleaning
    # ------------------------------------------------------------------

    def clean_plate(self, plate):
        """
        Refine the plate crop: adaptive threshold, find largest contour,
        validate ratio, return bounding coordinates.
        """
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)

            if not self.ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
                return plate, False, None

            return plate, True, [x, y, w, h]
        else:
            return plate, False, None

    # ------------------------------------------------------------------
    # Per-contour check
    # ------------------------------------------------------------------

    def check_plate(self, input_img, contour):
        """
        Validate a candidate contour as a plate, clean it, and
        segment characters. Returns (plate_img, chars, coords) or
        (None, None, None).
        """
        min_rect = cv2.minAreaRect(contour)

        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            roi = input_img[y: y + h, x: x + w]
            cleaned, plateFound, coordinates = self.clean_plate(roi)

            if plateFound:
                characters = segment_chars(cleaned, 400)
                if characters is not None and len(characters) == 8:
                    x1, y1, w1, h1 = coordinates
                    abs_coords = (x1 + x, y1 + y)
                    return cleaned, characters, abs_coords

        return None, None, None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def find_possible_plates(self, input_img):
        """
        Find all candidate license plates in *input_img*.

        Returns:
            list of plate image crops, or None if nothing found.
        Side effects:
            self.char_on_plate      – parallel list of character crops
            self.corresponding_area – parallel list of (x,y) coords
        """
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        contours = self.extract_contours(self.after_preprocess)

        for cnts in contours:
            plate, characters_on_plate, coordinates = self.check_plate(
                input_img, cnts
            )
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)

        return plates if len(plates) > 0 else None