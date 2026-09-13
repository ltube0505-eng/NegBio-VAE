import unittest

import torch

from distribution import NegBinomial


class NegativeBinomialKLTest(unittest.TestCase):
    def test_log_prob_matches_torch_negative_binomial(self):
        z = torch.tensor([0.0, 1.0, 3.0, 8.0])
        log_r = torch.tensor([-0.7, 0.0, 0.4, 1.2])
        logit_p = torch.tensor([-1.1, -0.2, 0.5, 1.4])

        r = torch.exp(log_r) + 1e-6
        expected = torch.distributions.NegativeBinomial(
            total_count=r,
            logits=-logit_p,
        ).log_prob(z)
        actual = NegBinomial._log_prob(z, log_r, logit_p)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_mc_kl_is_zero_when_prior_equals_posterior(self):
        dist = NegBinomial(
            reparam_type="gumbel",
            max_count=15,
            tau=0.7,
            num_samples=3,
        )
        log_r = torch.zeros(2, 4)
        logit_p = torch.zeros(2, 4)

        kl = dist.kl_mc(log_r, logit_p, log_r, logit_p)

        torch.testing.assert_close(kl, torch.zeros_like(kl), rtol=0, atol=0)

    def test_gamma_kl_matches_torch(self):
        alpha_q = torch.tensor([[0.8, 2.0, 4.5]])
        beta_q = torch.tensor([[0.5, 1.2, 3.0]])
        alpha_p = torch.tensor([[1.1, 1.5, 3.2]])
        beta_p = torch.tensor([[0.7, 0.9, 2.2]])
        expected = torch.distributions.kl_divergence(
            torch.distributions.Gamma(alpha_q, beta_q),
            torch.distributions.Gamma(alpha_p, beta_p),
        )
        actual = NegBinomial.gamma_kl(alpha_q, beta_q, alpha_p, beta_p)
        torch.testing.assert_close(actual, expected)

    def test_cch_kl_is_zero_when_prior_equals_posterior(self):
        dist = NegBinomial(reparam_type="gamma", tau=0.1, cts_max_count=64)
        log_alpha = torch.zeros(2, 4)
        log_beta = torch.zeros(2, 4)
        z_tilde = torch.rand(2, 4)

        kl = dist.kl_cch(
            log_alpha, log_beta, log_alpha, log_beta, z_tilde
        )

        torch.testing.assert_close(kl, torch.zeros_like(kl), rtol=0, atol=0)

    def test_cch_kl_has_finite_full_path_gradients(self):
        dist = NegBinomial(reparam_type="gamma", tau=0.1, cts_max_count=64)
        log_alpha_prior = torch.zeros(1, 3)
        log_beta_prior = torch.zeros(1, 3)
        log_alpha_post = torch.tensor(
            [[0.2, -0.1, 0.4]], requires_grad=True
        )
        log_beta_post = torch.tensor(
            [[-0.3, 0.2, 0.1]], requires_grad=True
        )
        z_tilde = dist.rsample(log_alpha_post, log_beta_post, t=0.1)

        kl = dist.kl_cch(
            log_alpha_prior,
            log_beta_prior,
            log_alpha_post,
            log_beta_post,
            z_tilde,
        ).sum()
        kl.backward()

        self.assertTrue(torch.isfinite(kl))
        self.assertTrue(torch.isfinite(log_alpha_post.grad).all())
        self.assertTrue(torch.isfinite(log_beta_post.grad).all())


if __name__ == "__main__":
    unittest.main()
