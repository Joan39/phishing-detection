import unittest

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


if __name__ == "__main__":
    unittest.main()
