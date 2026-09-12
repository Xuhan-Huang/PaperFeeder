from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase, IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from config import Config
from email_delivery import scheduled_delivery_time
from main import send_email, send_synthesis_failure_notification
from summarizer import SynthesisError


class DeliveryTimeTests(TestCase):
    def test_morning_utc_runner_schedules_same_beijing_day(self):
        target = scheduled_delivery_time("scheduled", now=datetime(2026, 9, 13, 1, 15, tzinfo=timezone.utc))
        self.assertEqual(target, datetime(2026, 9, 13, 3, 0, tzinfo=timezone.utc))

    def test_beijing_date_can_be_ahead_of_utc_date(self):
        target = scheduled_delivery_time("scheduled", now=datetime(2026, 9, 12, 17, 0, tzinfo=timezone.utc))
        self.assertEqual(target, datetime(2026, 9, 13, 3, 0, tzinfo=timezone.utc))

    def test_target_reached_or_passed_sends_now_not_next_day(self):
        for hour in (3, 4, 15):
            with self.subTest(hour=hour):
                self.assertIsNone(scheduled_delivery_time("scheduled", now=datetime(2026, 9, 13, hour, 0, tzinfo=timezone.utc)))

    def test_immediate_mode_never_schedules(self):
        self.assertIsNone(scheduled_delivery_time("immediate", now=datetime(2026, 9, 13, 1, 0, tzinfo=timezone.utc)))

    def test_custom_time_is_supported_by_config(self):
        target = scheduled_delivery_time("scheduled", "11:30", now=datetime(2026, 9, 13, 3, 0, tzinfo=timezone.utc))
        self.assertEqual(target, datetime(2026, 9, 13, 3, 30, tzinfo=timezone.utc))

    def test_invalid_settings_fail_early(self):
        for mode, target in (("tomorrow", "11:00"), ("scheduled", "25:00"), ("immediate", "11:99"), ("scheduled", 660)):
            with self.subTest(mode=mode, target=target), self.assertRaises(ValueError):
                Config(email_delivery_mode=mode, email_delivery_time=target)
        with self.assertRaisesRegex(ValueError, "timezone"):
            scheduled_delivery_time("scheduled", now=datetime(2026, 9, 13, 9, 0))

    def test_environment_overrides(self):
        with patch.dict("os.environ", {"EMAIL_DELIVERY_MODE": "immediate", "EMAIL_DELIVERY_TIME": "10:45"}):
            config = Config.from_yaml("missing-delivery-config.yaml")
        self.assertEqual(config.email_delivery_mode, "immediate")
        self.assertEqual(config.email_delivery_time, "10:45")


class DeliveryIntegrationTests(IsolatedAsyncioTestCase):
    async def test_scheduled_request_and_subject_use_beijing_date(self):
        sender = SimpleNamespace(send=AsyncMock(return_value=True))
        with patch("main.datetime") as clock, patch("main.ResendEmailer", return_value=sender), patch("builtins.print") as output:
            clock.now.return_value = datetime(2026, 9, 12, 17, 0, tzinfo=timezone.utc)
            result = await send_email("<p>Digest</p>", Config(email_to="owner@example.com"))
        self.assertTrue(result)
        self.assertEqual(sender.send.await_args.kwargs["scheduled_at"], "2026-09-13T03:00:00Z")
        self.assertIn("2026-09-13", sender.send.await_args.kwargs["subject"])
        self.assertIn("scheduled successfully", " ".join(str(call) for call in output.call_args_list))

    async def test_immediate_and_late_request_omit_scheduling(self):
        for mode, hour in (("immediate", 1), ("scheduled", 5)):
            sender = SimpleNamespace(send=AsyncMock(return_value=True))
            with self.subTest(mode=mode), patch("main.datetime") as clock, patch("main.ResendEmailer", return_value=sender):
                clock.now.return_value = datetime(2026, 9, 13, hour, 0, tzinfo=timezone.utc)
                await send_email("<p>Digest</p>", Config(email_delivery_mode=mode))
            self.assertIsNone(sender.send.await_args.kwargs["scheduled_at"])

    async def test_failure_notifications_are_immediate(self):
        sender = SimpleNamespace(send=AsyncMock(return_value=True))
        with patch("main.ResendEmailer", return_value=sender):
            await send_synthesis_failure_notification(SynthesisError("invalid JSON"), Config(email_delivery_mode="scheduled"))
        self.assertNotIn("scheduled_at", sender.send.await_args.kwargs)
