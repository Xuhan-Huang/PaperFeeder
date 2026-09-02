import unittest

from emailer import html_to_plain_text, sanitize_email_html


class EmailContentTest(unittest.TestCase):
    def test_sanitize_email_html_removes_scripts_and_preserves_links(self) -> None:
        source = """
        <html>
          <head>
            <script src="https://polyfill.example/script.js"></script>
            <script>window.alert('no')</script>
          </head>
          <body><a href="https://example.com/paper">Paper</a></body>
        </html>
        """

        sanitized = sanitize_email_html(source)

        self.assertNotIn("<script", sanitized.lower())
        self.assertNotIn("window.alert", sanitized)
        self.assertIn('href="https://example.com/paper"', sanitized)

    def test_html_to_plain_text_omits_styles_and_keeps_link_targets(self) -> None:
        source = """
        <html>
          <head><style>body { color: red; }</style></head>
          <body>
            <h1>Daily Digest</h1>
            <p>Read <a href="https://example.com/paper">this paper</a>.</p>
          </body>
        </html>
        """

        text = html_to_plain_text(source)

        self.assertIn("Daily Digest", text)
        self.assertIn("this paper (https://example.com/paper)", text)
        self.assertNotIn("color: red", text)


if __name__ == "__main__":
    unittest.main()
