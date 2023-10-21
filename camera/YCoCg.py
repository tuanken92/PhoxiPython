import cv2
import numpy as np

class YCoCg:
    # Subsampling of the chromatic channels Co, Cg.
    chromaSubsampling = 2

    # Pixel depth of B, G, R, and Y channels. Chromatic channels Co and Cg have an additional bit.
    pixelDepth = 10

    @staticmethod
    def pixelRGB(y, co, cg):
        zero_mask = y == 0
        delta = 1 << (YCoCg.pixelDepth - 1)
        maxValue = 2 * delta - 1
        r1 = 2 * y + co
        r = np.where(r1 > cg, (r1 - cg / 2), 0)
        g1 = y + cg / 2
        g = np.where(g1 > delta, (g1 - delta), 0)
        b1 = y + 2 * delta
        b2 = (co + cg) / 2
        b = np.where(b1 > b2, (b1 - b2), 0)
        return np.array([np.minimum(r, maxValue), np.minimum(g, maxValue), np.minimum(b, maxValue)]).T

    @staticmethod
    def convertToRGB(ycocgImg):
        yShift = np.uint16(16 - YCoCg.pixelDepth)
        mask = (1 << yShift) - 1
        rgbImg = np.zeros(ycocgImg.shape, dtype=np.uint16)

        for row in range(0, ycocgImg.shape[0], 2):
            for col in range(0, ycocgImg.shape[1], 2):
                y00 = ycocgImg[row, col, 0] >> yShift
                y01 = ycocgImg[row, col + 1, 0] >> yShift
                y10 = ycocgImg[row + 1, col, 0] >> yShift
                y11 = ycocgImg[row + 1, col + 1, 0] >> yShift
                co = ((ycocgImg[row, col, 1] & mask) << yShift) + (ycocgImg[row, col + 1, 1] & mask)
                cg = ((ycocgImg[row + 1, col, 2] & mask) << yShift) + (ycocgImg[row + 1, col + 1, 2] & mask)

                rgbImg[row, col, :] = YCoCg.pixelRGB(y00, co, cg)
                rgbImg[row, col + 1, :] = YCoCg.pixelRGB(y01, co, cg)
                rgbImg[row + 1, col, :] = YCoCg.pixelRGB(y10, co, cg)
                rgbImg[row + 1, col + 1, :] = YCoCg.pixelRGB(y11, co, cg)

        return rgbImg.astype(np.uint8)
