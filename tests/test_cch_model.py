import unittest

import torch
import torch.nn as nn

from model import GenericVAE


class CCHModelTest(unittest.TestCase):
    def test_cch_uses_two_residuals_and_reuses_soft_cts_sample(self):
        latent_dim = 3
        encoder = nn.Linear(4, 2 * latent_dim)
        decoder = nn.Identity()
        model = GenericVAE(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            dist_type="negbio",
            reparam_type="gamma",
            tau=0.1,
            cts_max_count=16,
            kl_type="cch",
        )
        with torch.no_grad():
            encoder.weight.zero_()
            encoder.bias.zero_()

        _, (log_alpha_q, log_beta_q), z_tilde, decoded = model(
            torch.zeros(2, 4)
        )

        torch.testing.assert_close(
            log_alpha_q, model.log_r_prior.expand_as(log_alpha_q)
        )
        torch.testing.assert_close(
            log_beta_q, model.logit_p_prior.expand_as(log_beta_q)
        )
        torch.testing.assert_close(decoded, z_tilde)
        self.assertEqual(z_tilde.shape, (2, latent_dim))
        self.assertTrue(((z_tilde % 1.0) != 0).any())

        z_prior, decoded_prior = model.sample_negative_binomial_prior(2)
        torch.testing.assert_close(decoded_prior, z_prior)
        self.assertEqual(z_prior.shape, (2, latent_dim))
        self.assertTrue(((z_prior % 1.0) != 0).any())

    def test_cch_rejects_gumbel_sampler(self):
        with self.assertRaisesRegex(ValueError, "reparam_type='gamma'"):
            GenericVAE(
                encoder=nn.Linear(4, 6),
                decoder=nn.Identity(),
                latent_dim=3,
                dist_type="negbio",
                reparam_type="gumbel",
                kl_type="cch",
            )


if __name__ == "__main__":
    unittest.main()
