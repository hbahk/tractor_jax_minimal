import unittest
import numpy as np
import jax.numpy as jnp
from tractor.jax.optimizer import optimize_fluxes, extract_model_data, render_scene
from tractor.image import Image
from tractor.pointsource import PointSource
from tractor.wcs import NullWCS, PixPos
from tractor.brightness import Flux
from tractor.psf import GaussianMixturePSF
from tractor.engine import Tractor

class ExampleMultiImageFitting(unittest.TestCase):
    def test_multi_exposure_fitting(self):
        """
        Demonstrates fitting fluxes of sources across multiple overlapping images (exposures).
        """
        print("Running Multi-Image Fitting Example...")

        # 1. Define Scene Parameters
        H, W = 30, 30
        sigma = 1.5
        # Simple Gaussian PSF
        psf = GaussianMixturePSF(
            [1.0],           # amplitude
            [[0.0, 0.0]],    # mean
            [[[sigma**2, 0.0], [0.0, sigma**2]]] # covariance
        )

        # 2. Define Sources (Truth)
        true_flux1 = 500.0
        true_flux2 = 300.0

        # Overlapping sources
        src1_pos = PixPos(15.0, 15.0)
        src2_pos = PixPos(18.0, 15.0) # 3 pixels away

        src1 = PointSource(src1_pos, Flux(true_flux1))
        src2 = PointSource(src2_pos, Flux(true_flux2))

        catalog = [src1, src2]

        # 3. Create Images (Dithered exposures)
        # We use NullWCS with offsets to simulate dithering.
        # Image 1: Centered
        wcs1 = NullWCS(dx=0.0, dy=0.0)

        # Image 2: Shifted by (0.5, 0.5) pixels
        # NullWCS(dx, dy) means pixel (x,y) -> pos (x-dx, y-dy).
        # So if we want the camera to be shifted by (0.5, 0.5),
        # the source at (15, 15) should appear at (14.5, 14.5).
        # pixel = pos - (-dx) ? No.
        # NullWCS: pos = pixel - dx. => pixel = pos + dx.
        # If we want source at 15 to be at 14.5, then 14.5 = 15 + dx => dx = -0.5.
        wcs2 = NullWCS(dx=-0.5, dy=-0.5)

        # Image 3: Shifted by (-0.5, 0.0)
        wcs3 = NullWCS(dx=0.5, dy=0.0)

        # Create dummy images first to use render_scene for generation
        dummy_data = np.zeros((H, W))
        dummy_invvar = np.ones((H, W)) # Uniform weighting

        img1 = Image(data=dummy_data, inverr=np.sqrt(dummy_invvar), psf=psf, wcs=wcs1, name="Img1")
        img2 = Image(data=dummy_data, inverr=np.sqrt(dummy_invvar), psf=psf, wcs=wcs2, name="Img2")
        img3 = Image(data=dummy_data, inverr=np.sqrt(dummy_invvar), psf=psf, wcs=wcs3, name="Img3")

        images = [img1, img2, img3]
        tractor_true = Tractor(images, catalog)

        # 4. Generate Synthetic Data (Truth)
        # Use JAX rendering to ensure model consistency
        print("Generating synthetic data...")
        images_data, batches, true_fluxes_jax = extract_model_data(tractor_true)
        model_images = render_scene(true_fluxes_jax, images_data, batches)

        # Add noise and update images
        np.random.seed(42)
        noise_level = 1.0

        images_with_data = []
        for i, img in enumerate(images):
            # Get rendered model
            model = np.array(model_images[i])
            # Add noise
            noise = np.random.normal(0, noise_level, size=(H, W))
            noisy_data = model + noise

            # Create new Image object with data
            new_img = Image(
                data=noisy_data,
                inverr=np.ones((H, W)) / noise_level,
                psf=psf,
                wcs=img.getWcs(),
                name=img.name
            )
            images_with_data.append(new_img)

        # 5. Setup Optimization (Perturbed fluxes)
        print("Setting up optimization...")
        # Perturb fluxes
        src1_perturbed = PointSource(src1_pos, Flux(true_flux1 * 0.5)) # Start far off
        src2_perturbed = PointSource(src2_pos, Flux(true_flux2 * 1.5))

        catalog_perturbed = [src1_perturbed, src2_perturbed]

        tractor_opt = Tractor(images_with_data, catalog_perturbed)

        # Check initial fluxes
        print(f"Initial Fluxes: {[s.brightness.getValue() for s in catalog_perturbed]}")

        # 6. Run Optimization
        print("Optimizing...")
        tractor_opt = optimize_fluxes(tractor_opt)

        # 7. Verify Results
        opt_flux1 = catalog_perturbed[0].brightness.getValue()
        opt_flux2 = catalog_perturbed[1].brightness.getValue()

        print(f"Optimized Fluxes: [{opt_flux1:.2f}, {opt_flux2:.2f}]")
        print(f"True Fluxes:      [{true_flux1:.2f}, {true_flux2:.2f}]")

        # Tolerances (approximate Fisher info estimation)
        # Total S/N is high.
        # Flux ~ 500. Peak pixel ~ 500 / (2pi * 1.5^2) ~ 35.
        # Noise = 1.
        # 3 images.
        # Should be very accurate.

        diff1 = abs(opt_flux1 - true_flux1)
        diff2 = abs(opt_flux2 - true_flux2)

        print(f"Diffs: {diff1:.2f}, {diff2:.2f}")

        self.assertTrue(diff1 < 10.0, f"Source 1 flux error too large: {diff1}")
        self.assertTrue(diff2 < 10.0, f"Source 2 flux error too large: {diff2}")

        print("Multi-image fitting successful!")

if __name__ == "__main__":
    unittest.main()
