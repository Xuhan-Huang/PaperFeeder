import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from emailer import ResendEmailer, html_to_plain_text, sanitize_email_html


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


class ResendDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def submit(self, *, scheduled_at=None, status=200, response_body=None):
        response = MagicMock(status=status)
        response.json = AsyncMock(return_value={"id": "email-test"} if response_body is None else response_body)
        response.text = AsyncMock(return_value="request rejected")
        response.__aenter__ = AsyncMock(return_value=response)
        session = MagicMock()
        session.post.return_value = response
        session.__aenter__ = AsyncMock(return_value=session)
        with patch("emailer.aiohttp.ClientSession", return_value=session):
            success = await ResendEmailer("test").send(
                "owner@example.com", "Digest", '<script>bad()</script><a href="https://arxiv.org/abs/test">Paper</a>',
                attachments=[{"filename": "data.json", "content": "e30="}], scheduled_at=scheduled_at,
            )
        return success, session.post.call_args.kwargs["json"]

    async def test_scheduled_payload_retains_safe_content_and_attachments(self):
        success, payload = await self.submit(scheduled_at="2026-09-13T03:00:00Z")
        self.assertTrue(success)
        self.assertEqual(payload["scheduled_at"], "2026-09-13T03:00:00Z")
        self.assertNotIn("<script", payload["html"])
        self.assertIn("https://arxiv.org/abs/test", payload["text"])
        self.assertEqual(len(payload["attachments"]), 1)

    async def test_immediate_payload_has_no_scheduled_at(self):
        success, payload = await self.submit()
        self.assertTrue(success)
        self.assertNotIn("scheduled_at", payload)

    async def test_rejection_or_missing_email_id_is_not_success(self):
        success, _ = await self.submit(status=422)
        self.assertFalse(success)
        success, _ = await self.submit(response_body={})
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()
