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


if __name__ == "__main__":
    unittest.main()
