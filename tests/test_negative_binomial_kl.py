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

    def test_mc_kl_samples_and_scores_with_posterior_dispersion(self):
        class FixedSampler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_log_r = []

            def forward(self, log_r, logit_p, hard=False):
                self.seen_log_r.append(log_r.detach().clone())
                return torch.ones_like(log_r)

        dist = NegBinomial(reparam_type="gumbel", num_samples=2)
        sampler = FixedSampler()
        dist.strategy = sampler
        log_r_prior = torch.zeros(1, 3)
        log_r_post = torch.full((1, 3), 0.7)
        logit_p_prior = torch.full((1, 3), -0.2)
        logit_p_post = torch.full((1, 3), 0.4)

        actual = dist.kl_mc(
            log_r_prior, logit_p_prior, log_r_post, logit_p_post
        )
        samples = torch.ones_like(log_r_post)
        expected = (
            NegBinomial._log_prob(samples, log_r_post, logit_p_post)
            - NegBinomial._log_prob(samples, log_r_prior, logit_p_prior)
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(sampler.seen_log_r), 2)
        for seen in sampler.seen_log_r:
            torch.testing.assert_close(seen, log_r_post)

if __name__ == "__main__":
    unittest.main()
