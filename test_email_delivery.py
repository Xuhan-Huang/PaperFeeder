from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from config import Config
from main import send_email, send_synthesis_failure_notification
from summarizer import SynthesisError


class DeliveryIntegrationTests(IsolatedAsyncioTestCase):
    async def test_immediate_request_subject_uses_beijing_date(self):
        sender = SimpleNamespace(send=AsyncMock(return_value=True))
        with patch("main.datetime") as clock, patch("main.ResendEmailer", return_value=sender), patch("builtins.print") as output:
            clock.now.return_value = datetime(2026, 9, 12, 17, 0, tzinfo=timezone.utc)
            result = await send_email("<p>Digest</p>", Config(email_to="owner@example.com"))
        self.assertTrue(result)
        self.assertNotIn("scheduled_at", sender.send.await_args.kwargs)
        self.assertIn("2026-09-13", sender.send.await_args.kwargs["subject"])
        self.assertIn("inbox delivery is not yet confirmed", " ".join(str(call) for call in output.call_args_list))

    async def test_morning_and_afternoon_requests_are_immediate(self):
        for hour in (1, 5):
            sender = SimpleNamespace(send=AsyncMock(return_value=True))
            with self.subTest(hour=hour), patch("main.datetime") as clock, patch("main.ResendEmailer", return_value=sender):
                clock.now.return_value = datetime(2026, 9, 13, hour, 0, tzinfo=timezone.utc)
                await send_email("<p>Digest</p>", Config())
            self.assertNotIn("scheduled_at", sender.send.await_args.kwargs)

    async def test_failure_notifications_are_immediate(self):
        sender = SimpleNamespace(send=AsyncMock(return_value=True))
        with patch("main.ResendEmailer", return_value=sender):
            await send_synthesis_failure_notification(SynthesisError("invalid JSON"), Config())
        self.assertNotIn("scheduled_at", sender.send.await_args.kwargs)
