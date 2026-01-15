
import unittest
import numpy as np

try:
    from astropy.wcs import WCS
    has_astropy = True
except ImportError:
    has_astropy = False

from tractor.wcs import AstropyWCS, RaDecPos

class TestAstropyWCSPadding(unittest.TestCase):
    def test_padding_shift(self):
        """
        Verify that RaDecPos correctly tracks the source position when the
        AstropyWCS is shifted (simulating image padding).
        """
        if not has_astropy:
            print("Skipping test because astropy is missing")
            return

        # Create a simple WCS
        # 1 arcsec/pixel, centered at (100, 100) -> RA=10, Dec=10
        w = WCS(naxis=2)
        w.wcs.crpix = [100, 100]
        w.wcs.cdelt = [-1./3600, 1./3600]
        w.wcs.crval = [10, 10]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Tractor AstropyWCS wrapper
        # origin=0 means 0-based indexing (numpy style)
        awcs = AstropyWCS(w, origin=0)

        # Define a source position at the center
        # RaDecPos takes positional arguments (ra, dec)
        src_pos = RaDecPos(10.0, 10.0)

        # Get pixel position in original WCS
        x0, y0 = awcs.positionToPixel(src_pos)

        # Verify initial expectation (should be near 99, 99 for 0-based indexing if CRPIX is 100,100 1-based)
        # 100 1-based = 99 0-based.
        self.assertAlmostEqual(x0, 99.0, places=3)
        self.assertAlmostEqual(y0, 99.0, places=3)

        # Simulate padding the image by adding 10 pixels to left and bottom.
        # This shifts the image origin by (-10, -10).
        pad_x = 10
        pad_y = 10

        # If we pad, we must shift the WCS by -pad to maintain alignment.
        new_awcs = awcs.shifted(-pad_x, -pad_y)

        # Get new pixel position
        x1, y1 = new_awcs.positionToPixel(src_pos)

        # We expect the pixel coordinates to increase by the padding amount.
        self.assertAlmostEqual(x1, x0 + pad_x, places=5)
        self.assertAlmostEqual(y1, y0 + pad_y, places=5)

if __name__ == '__main__':
    unittest.main()
