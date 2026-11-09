import unittest

import pandas as pd

from backend.url_features import extract_url_features


class UrlFeatureTests(unittest.TestCase):
    def test_extracts_expected_features_from_https_url(self):
        features = extract_url_features(
            "https://www.example.com/login?user=alice&id=42"
        )

        self.assertEqual(features.shape, (1, 19))
        self.assertEqual(int(features.at[0, "IsHTTPS"]), 1)
        self.assertEqual(int(features.at[0, "NoOfQMarkInURL"]), 1)
        self.assertEqual(int(features.at[0, "NoOfEqualsInURL"]), 2)
        self.assertEqual(int(features.at[0, "NoOfDegitsInURL"]), 2)
        self.assertEqual(int(features.at[0, "NoOfSubDomain"]), 1)
        self.assertEqual(int(features.at[0, "IsTrustedRoot"]), 0)

    def test_accepts_url_without_scheme(self):
        features = extract_url_features("example.com/path")

        self.assertEqual(int(features.at[0, "IsHTTPS"]), 0)
        self.assertEqual(int(features.at[0, "DomainLength"]), len("example.com"))

    def test_rejects_empty_or_invalid_url(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            extract_url_features(" ")

        with self.assertRaisesRegex(ValueError, "invalid host or port"):
            extract_url_features("https://example.com:invalid/")

        with self.assertRaisesRegex(ValueError, "whitespace"):
            extract_url_features("https://google .com")

    def test_marks_only_clean_trusted_roots(self):
        self.assertEqual(
            int(extract_url_features("https://google.com").at[0, "IsTrustedRoot"]),
            1,
        )
        self.assertEqual(
            int(
                extract_url_features("https://google.com/login").at[
                    0, "IsTrustedRoot"
                ]
            ),
            0,
        )

    def test_training_policy_includes_canonical_legitimate_pages(self):
        from backend.domain_policy import trusted_root_training_urls

        urls = trusted_root_training_urls()

        self.assertIn("https://accounts.google.com", urls)
        self.assertIn("https://www.wikipedia.org/wiki/Main_Page", urls)

    def test_trusted_domain_matching_requires_a_host_boundary(self):
        from backend.domain_policy import is_trusted_domain_url

        self.assertTrue(is_trusted_domain_url("https://accounts.google.com"))
        self.assertFalse(is_trusted_domain_url("https://google.com.evil.test"))

    def test_calibration_cap_needs_trusted_host_and_clean_shape(self):
        from backend.url_features import trusted_calibration_applies

        clean = "https://en.wikipedia.org/wiki/Main_Page"
        self.assertTrue(
            trusted_calibration_applies(clean, extract_url_features(clean))
        )
        # Encoded obfuscation on a trusted host is not the real brand.
        rigged = "https://en.wikipedia.org/%2e%2e/verify?x=%40"
        self.assertFalse(
            trusted_calibration_applies(rigged, extract_url_features(rigged))
        )
        lookalike = "https://en.wikipedia.org.login-verify-account.xyz"
        self.assertFalse(
            trusted_calibration_applies(
                lookalike, extract_url_features(lookalike)
            )
        )

    def test_calibration_cap_refuses_an_open_redirect_on_a_brand_host(self):
        from backend.url_features import trusted_calibration_applies

        jump = "https://www.google.com/url?sa=t&url=https://refund-claim-irs-com/tax"
        self.assertFalse(
            trusted_calibration_applies(jump, extract_url_features(jump))
        )

    def test_redirect_target_extraction(self):
        from backend.url_features import redirect_targets

        # The two shapes that bypassed the deployed models.
        self.assertEqual(
            redirect_targets(
                "https://www.google.com/url?sa=t&url=https://refund-claim-irs-com/tax"
            ),
            ["https://refund-claim-irs-com/tax"],
        )
        self.assertEqual(
            redirect_targets(
                "https://l.facebook.com.l.php?u=https://secure-appleid-verify.com/login"
            ),
            ["https://secure-appleid-verify.com/login"],
        )
        # Percent-encoded and protocol-relative payloads are decoded.
        self.assertEqual(
            redirect_targets(
                "https://login.example.test/sso?redirect_uri=https%3A%2F%2Fevil.test%2Fcb"
            ),
            ["https://evil.test/cb"],
        )
        self.assertEqual(
            redirect_targets("https://brand.example.test/go?url=//evil.test/x"),
            ["https://evil.test/x"],
        )

    def test_same_brand_redirect_is_not_a_payload(self):
        from backend.url_features import redirect_targets

        self.assertEqual(
            redirect_targets(
                "https://accounts.google.com/Login?continue=https://mail.google.com/mail/"
            ),
            [],
        )

    def test_non_url_and_unknown_parameter_values_are_ignored(self):
        from backend.url_features import redirect_targets

        for url in (
            "https://www.google.com/search?q=url&btnK=Search",
            "https://news.ycombinator.com/item?id=1&url=/local/path",
            "https://brand.example.test/go?url=javascript:alert(1)",
            "https://brand.example.test/page",
            "https://brand.example.test/go?next=data:text/html,<script>",
        ):
            self.assertEqual(redirect_targets(url), [], url)


    def test_batch_extraction_matches_row_by_row(self):
        from backend.url_features import extract_url_features_frame

        urls = [
            "https://example.com/a?b=1",
            "http://192.168.0.1/x",
            "no-scheme.test/path",
            "https://en.wikipedia.org/wiki/Main_Page",
        ]

        pd.testing.assert_frame_equal(
            extract_url_features_frame(urls),
            pd.concat(
                [extract_url_features(url) for url in urls], ignore_index=True
            ),
        )

    def test_general_negatives_do_not_use_battery_hosts(self):
        """Training counter-examples must stay disjoint from the test battery."""
        from backend.domain_policy import general_legitimate_training_urls
        from scripts.realworld_battery import LEGITIMATE_URLS
        from urllib.parse import urlsplit

        def hosts(urls):
            return {
                (urlsplit(url if "://" in url else "//" + url).hostname or "")
                .lower()
                .removeprefix("www.")
                for url in urls
            }

        self.assertFalse(
            hosts(general_legitimate_training_urls()) & hosts(LEGITIMATE_URLS)
        )

    def test_general_negatives_are_disjoint_from_fresh_probe_hosts(self):
        from backend.domain_policy import legitimate_training_urls
        from scripts.fresh_probe import FRESH_LEGITIMATE, FRESH_PHISHING
        from urllib.parse import urlsplit

        trained = {
            urlsplit(url).hostname.lower().removeprefix("www.")
            for url in legitimate_training_urls()
        }
        probe = {
            urlsplit(url).hostname.lower().removeprefix("www.")
            for url in FRESH_LEGITIMATE + FRESH_PHISHING
        }
        # Only the brand roots the trusted-domain cap exists for may coincide.
        self.assertLessEqual(probe & trained, {"wikipedia.org", "github.com"})

    def test_negatives_cover_urls_longer_than_any_dataset_legitimate_row(self):
        """PhiUSIIL legitimate rows top out around 39 characters (p99)."""
        from backend.domain_policy import general_legitimate_training_urls
        from backend.url_features import extract_url_features_frame

        lengths = extract_url_features_frame(
            general_legitimate_training_urls()
        )["URLLength"]
        self.assertGreater(float((lengths > 50).mean()), 0.2)

    def test_hard_negative_blocks_weight_bare_http_separately(self):
        from backend.domain_policy import (
            BARE_HTTP_REPLICAS,
            HARD_NEGATIVE_REPLICAS,
            bare_http_origin_training_urls,
            hard_negative_blocks,
        )

        blocks = hard_negative_blocks()

        self.assertEqual(
            [replicas for _, replicas in blocks],
            [HARD_NEGATIVE_REPLICAS, BARE_HTTP_REPLICAS],
        )
        self.assertEqual(
            blocks[1][0],
            bare_http_origin_training_urls(),
            "the uplift block must be exactly the bare origins, no paths",
        )
        from urllib.parse import urlsplit

        self.assertTrue(
            all(urlsplit(url).path in ("", "/") for url in blocks[1][0]),
            blocks[1][0][:3],
        )

    def test_disabling_general_hosts_drops_the_bare_http_block(self):
        from backend.domain_policy import hard_negative_blocks

        blocks = hard_negative_blocks(include_general_hosts=False)

        self.assertEqual(len(blocks), 1)


if __name__ == "__main__":
    unittest.main()
