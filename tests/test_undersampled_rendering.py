
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from tractor import Tractor, Image, Catalog, PointSource, PixelizedPSF, GaussianMixturePSF, ConstantSky, Flux, NullWCS, PixPos
from tractor.galaxy import ExpGalaxy, GalaxyShape, JaxGalaxy
from tractor.jax.optimizer import extract_model_data, optimize_fluxes, render_image
from tractor import mixture_profiles as mp

class TestUndersampledRendering(unittest.TestCase):
    def test_anisotropic_undersampled(self):
        print("\n--- Test Undersampled Anisotropic Rendering ---")

        # 1. Setup Scene
        # Small image, coarse pixels
        H, W = 20, 20
        data = np.zeros((H, W))
        invvar = np.ones((H, W))
        inverr = np.sqrt(invvar)

        # 2. Create Anisotropic Gaussian PSF
        # Define Gaussian parameters
        # Sigma = [0.3, 0.1] pixels (very small, anisotropic)
        # In variance: sigma^2
        sigma_x = 0.3
        sigma_y = 0.1
        var_x = sigma_x**2
        var_y = sigma_y**2

        # Create GaussianMixturePSF (Analytic)
        amp = np.array([1.0])
        mean = np.zeros((1, 2))
        var = np.array([[[var_x, 0.0], [0.0, var_y]]])

        psf_mog = GaussianMixturePSF(amp, mean, var)

        # Create PixelizedPSF (Oversampled)
        # Sampling = 0.2 (5x oversampling)
        sampling = 0.2

        # Render PSF onto fine grid
        # Fine pixel size = 0.2 image pixels
        # Grid size: enough to cover PSF
        # 5 sigma ~ 1.5 pixels.
        # Fine grid size ~ 1.5 / 0.2 ~ 7.5 pixels radius.
        # Let's make it 21x21 fine pixels.
        ph_fine, pw_fine = 21, 21

        # Evaluate MoG on fine grid
        # Grid coordinates in image pixels
        # Center is at (0, 0) relative to PSF center

        # Pixel coordinates on fine grid
        # Center index
        cx, cy = pw_fine // 2, ph_fine // 2

        # Generate grid
        # y, x indices
        y_idx, x_idx = np.indices((ph_fine, pw_fine))

        # Convert to offsets in image pixels
        # pixel (i, j) corresponds to offset ( (j - cx)*sampling, (i - cy)*sampling )
        y_off = (y_idx - cy) * sampling
        x_off = (x_idx - cx) * sampling

        # Evaluate Gaussian
        # 1 / (2pi sqrt(det)) * exp(-0.5 * r^T S^-1 r)
        det = var_x * var_y
        inv_var_x = 1.0 / var_x
        inv_var_y = 1.0 / var_y

        norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
        exponent = -0.5 * (x_off**2 * inv_var_x + y_off**2 * inv_var_y)
        psf_fine_img = norm * np.exp(exponent)

        # Normalize sum
        psf_fine_img /= np.sum(psf_fine_img)

        psf_pix = PixelizedPSF(psf_fine_img)
        psf_pix.sampling = sampling

        # 3. Create Sheared Galaxy
        # Pos: Center of image
        pos_x, pos_y = 10.0, 10.0
        flux_val = 1000.0

        # Shape: re=0.5 pixels (small), ab=0.5 (elliptical), phi=45 (rotated)
        # Note: re is effective radius. For Exp, half-light radius.
        # GalaxyShape params order: re, ab, phi
        shape = GalaxyShape(0.5, 0.5, 45.0)

        # Source 1: Using MoG PSF (Analytic)
        src1 = ExpGalaxy(PixPos(pos_x, pos_y), Flux(flux_val), shape)
        img1 = Image(data=data, inverr=inverr, psf=psf_mog, wcs=NullWCS(), sky=ConstantSky(0.0))
        tractor1 = Tractor([img1], [src1])

        # Source 2: Using Pixelized PSF (FFT, Oversampled)
        src2 = ExpGalaxy(PixPos(pos_x, pos_y), Flux(flux_val), shape)
        img2 = Image(data=data, inverr=inverr, psf=psf_pix, wcs=NullWCS(), sky=ConstantSky(0.0))
        tractor2 = Tractor([img2], [src2])

        # 4. Render
        print("Rendering Analytic (MoG)...")
        # For MoG, extract_model_data will create MoG batches and render using render_galaxy_mog
        images_data1, batches1, fluxes1 = extract_model_data(tractor1, oversample_rendering=False)
        # Need to slice batches for render_image
        batch1_single = self.slice_batch(batches1, 0)
        model1 = render_image(fluxes1[0], jax.tree_util.tree_map(lambda x: x[0], images_data1), batch1_single)

        print("Rendering FFT (Pixelized, Oversampled)...")
        # For Pixelized, extract_model_data will create FFT batches.
        # sampling=0.2 means max_factor=5.
        # It should use high-res grid.
        images_data2, batches2, fluxes2 = extract_model_data(tractor2, oversample_rendering=True)
        batch2_single = self.slice_batch(batches2, 0)
        model2 = render_image(fluxes2[0], jax.tree_util.tree_map(lambda x: x[0], images_data2), batch2_single)

        # 5. Compare
        # Compare images
        print(f"Model1 shape: {model1.shape}")
        print(f"Model2 shape: {model2.shape}")

        # If model2 is padded (due to FFT padding), crop it
        if model2.shape != model1.shape:
            H, W = model1.shape
            # The padding is added to the end (right/bottom)
            model2 = model2[:H, :W]

        sum1 = np.sum(model1)
        sum2 = np.sum(model2)

        print(f"Flux Analytic: {sum1}")
        print(f"Flux FFT: {sum2}")
        print(f"Expected Flux: {flux_val}")

        # Visual check (print small patch)
        cx, cy = int(pos_x), int(pos_y)
        sl = (slice(cy-2, cy+3), slice(cx-2, cx+3))
        print("\nAnalytic Patch:")
        print(model1[sl])
        print("\nFFT Patch:")
        print(model2[sl])

        # Validation
        # The FFT model should conserve flux significantly better than the point-sampled analytic model
        # for this undersampled case.

        # Check FFT flux conservation
        # Allow 1% error
        rel_err_fft = abs(sum2 - flux_val) / flux_val
        print(f"FFT Relative Flux Error: {rel_err_fft}")
        self.assertTrue(rel_err_fft < 0.01, f"FFT Flux not conserved: {sum2} vs {flux_val}")

        # Check Peak Position
        # Should be at (10, 10)
        peak_idx = np.unravel_index(np.argmax(model2), model2.shape)
        print(f"Peak Location: {peak_idx}")
        self.assertEqual(peak_idx, (10, 10), "Peak location incorrect")

    def slice_batch(self, batches, idx):
        single = {}
        for k, v in batches.items():
            single[k] = {}
            for subk, subv in v.items():
                if subk in ['pos_pix', 'wcs_cd_inv']:
                    single[k][subk] = subv[idx]
                else:
                    single[k][subk] = subv # shared
        return single

if __name__ == '__main__':
    unittest.main()
