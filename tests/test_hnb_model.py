import unittest

import torch

from hnb_model import HNBVAE


class HNBModelTest(unittest.TestCase):
    def _model(self, kl_type="mc", reparam_type="gamma"):
        return HNBVAE(
            input_channels=1,
            image_size=16,
            groups_per_scale=[1, 2],
            channels_per_scale=[16, 8],
            latent_channels_per_scale=[4, 3],
            kl_type=kl_type,
            reparam_type=reparam_type,
            tau=1.0,
            num_samples=3,
            spatial_sizes=[4, 8],
        )

    def test_multiscale_forward_shapes_and_neutral_posterior(self):
        model = self._model()
        x = torch.rand(2, 1, 16, 16)

        output = model(x)

        self.assertEqual(output.reconstruction.shape, x.shape)
        self.assertEqual(len(output.latents), 3)
        self.assertEqual(output.latents[0].shape, (2, 4, 4, 4))
        self.assertEqual(output.latents[1].shape, (2, 3, 8, 8))
        self.assertEqual(output.latents[2].shape, (2, 3, 8, 8))
        self.assertEqual(output.latent.shape, (2, 10))
        self.assertEqual(output.representation.shape, (2, 10))
        self.assertEqual(output.kl_diag.shape, (2, 10))
        self.assertEqual(output.kl_per_group.shape, (2, 3))
        self.assertEqual(output.total_kl.shape, (2,))
        self.assertLess(output.total_kl.abs().max().item(), 1e-4)

        for params in output.parameters:
            torch.testing.assert_close(params["alpha_q"], params["alpha_p"])
            torch.testing.assert_close(params["beta_q"], params["beta_p"])

    def test_forward_backward_has_finite_gradients(self):
        model = self._model()
        x = torch.rand(2, 1, 16, 16)

        output = model(x)
        loss = model.mse_loss(x, output.reconstruction) + output.total_kl.mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
        self.assertTrue(any(
            ((latent % 1.0) != 0).any().item()
            for latent in output.latents
        ))

    def test_prior_generation_runs_top_down_without_encoder(self):
        model = self._model()

        latent, samples = model.sample_prior(3)

        self.assertEqual(latent.shape, (3, 10))
        self.assertEqual(samples.shape, (3, 1, 16, 16))
        self.assertTrue(torch.isfinite(latent).all())
        self.assertTrue(torch.isfinite(samples).all())

    def test_ladder_warmup_activates_groups_from_top_to_bottom(self):
        model = self._model()
        group_kl = torch.ones(2, 3)

        at_start = model.training_kl(group_kl, warmup_progress=0.0)
        after_first_group = model.training_kl(
            group_kl, warmup_progress=1.0 / 3.0
        )
        at_end = model.training_kl(group_kl, warmup_progress=1.0)

        torch.testing.assert_close(at_start, torch.zeros(2))
        torch.testing.assert_close(after_first_group, torch.ones(2))
        torch.testing.assert_close(at_end, torch.full((2,), 3.0))

    def test_analytical_nb_kl_is_rejected_for_unshared_dispersion(self):
        with self.assertRaisesRegex(ValueError, "requires kl='mc'"):
            self._model("analytical")

    def test_non_mc_objective_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires kl='mc'"):
            self._model("gamma")

    def test_gumbel_sampler_runs_and_temperature_updates(self):
        model = self._model(reparam_type="gumbel")
        model.set_temperature(0.25)

        output = model(torch.rand(2, 1, 16, 16))

        self.assertEqual(output.reconstruction.shape, (2, 1, 16, 16))
        self.assertTrue(torch.isfinite(output.total_kl).all())
        for layer in model.td_layers:
            self.assertEqual(layer.relaxation.dist.strategy.tau, 0.25)


if __name__ == "__main__":
    unittest.main()
