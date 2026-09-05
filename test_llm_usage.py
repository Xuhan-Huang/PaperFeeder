from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from config import Config
from llm_client import LLMClient, LLMResult, LLMUsage, normalize_usage
from summarizer import PaperSummarizer


def make_result(text: str, prompt_tokens: int = 10, completion_tokens: int = 4) -> LLMResult:
    return LLMResult(
        text=text,
        usage=LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_tokens=2,
            reasoning_tokens_reported=1,
            usage_source="test",
            available=True,
        ),
        finish_reason="stop",
    )


class LLMUsageClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_stream_captures_final_usage_and_effort(self) -> None:
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"), finish_reason=None)],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"), finish_reason="stop")],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "total_tokens": 27,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                    "usage_source": "anthropic",
                },
            ),
        ]

        class FakeStream:
            def __aiter__(self):
                self._iterator = iter(chunks)
                return self

            async def __anext__(self):
                try:
                    return next(self._iterator)
                except StopIteration as error:
                    raise StopAsyncIteration from error

        client = LLMClient(api_key="test", base_url="https://example.com/v1", max_retries=0)
        create = AsyncMock(return_value=FakeStream())
        client.async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        result = await client.achat_stream_with_usage(
            [{"role": "user", "content": "test"}],
            max_tokens=10,
            reasoning_effort="low",
        )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertEqual(result.usage.completion_tokens, 7)
        self.assertEqual(result.usage.cached_tokens, 3)
        self.assertEqual(result.usage.reasoning_tokens_reported, 2)
        self.assertEqual(result.usage.usage_source, "anthropic")
        self.assertTrue(result.usage.available)
        self.assertEqual(create.await_args.kwargs["stream_options"], {"include_usage": True})
        self.assertEqual(create.await_args.kwargs["reasoning_effort"], "low")

    async def test_openai_nonstream_captures_usage(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=5,
                total_tokens=17,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
                usage_source="openai",
            ),
        )
        client = LLMClient(api_key="test", base_url="https://example.com/v1", max_retries=0)
        create = AsyncMock(return_value=response)
        client.async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        result = await client.achat_with_usage(
            [{"role": "user", "content": "test"}],
            reasoning_effort="medium",
        )

        self.assertEqual(result.text, "done")
        self.assertEqual(result.usage.total_tokens, 17)
        self.assertEqual(result.usage.cached_tokens, 4)
        self.assertEqual(create.await_args.kwargs["reasoning_effort"], "medium")

    async def test_missing_usage_is_explicit_and_does_not_fail_text(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="done"), finish_reason="stop")
            ],
            usage=None,
        )
        client = LLMClient(api_key="test", base_url="https://example.com/v1", max_retries=0)
        client.async_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=response))
            )
        )

        result = await client.achat_with_usage([{"role": "user", "content": "test"}])

        self.assertEqual(result.text, "done")
        self.assertFalse(result.usage.available)
        self.assertEqual(normalize_usage(None), LLMUsage())

    def test_anthropic_usage_fields_are_normalized(self) -> None:
        usage = normalize_usage(
            {
                "input_tokens": 30,
                "output_tokens": 8,
                "cache_read_input_tokens": 6,
            }
        )
        self.assertEqual(usage.prompt_tokens, 30)
        self.assertEqual(usage.completion_tokens, 8)
        self.assertEqual(usage.total_tokens, 38)
        self.assertEqual(usage.cached_tokens, 6)


class UsageLedgerTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_attempts_are_preserved_and_artifact_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summarizer = PaperSummarizer(
                api_key="TOP_SECRET_API_KEY",
                base_url="https://example.com/v1",
                model="anthropic/claude-opus-5",
                synthesis_reasoning_effort="low",
                diagnostic_output_dir=directory,
            )
            await asyncio.gather(
                *(
                    summarizer._record_usage_attempt(
                        purpose=f"adaptive compaction p{index:02d}",
                        attempt=1,
                        status="success",
                        elapsed_seconds=0.1,
                        result=make_result(
                            f"TOP_SECRET_RESPONSE_{index}",
                            prompt_tokens=index,
                            completion_tokens=2,
                        ),
                    )
                    for index in range(1, 21)
                )
            )
            path = await summarizer._write_usage_report("run-test")

            self.assertIsNotNone(path)
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(len(payload["attempts"]), 20)
            self.assertEqual(payload["totals"]["requests"], 20)
            self.assertEqual(payload["totals"]["prompt_tokens"], sum(range(1, 21)))
            artifact_text = Path(path).read_text(encoding="utf-8")
            self.assertNotIn("TOP_SECRET_RESPONSE", artifact_text)
            self.assertNotIn("TOP_SECRET_API_KEY", artifact_text)
            self.assertNotIn("paper content", artifact_text.lower())

    async def test_retry_ledger_counts_failed_and_successful_attempts(self) -> None:
        summarizer = PaperSummarizer(
            api_key="test",
            base_url="https://example.com/v1",
            synthesis_retries=1,
            synthesis_retry_base_delay_sec=0,
        )
        summarizer.client.achat_stream_with_usage = AsyncMock(
            side_effect=[TimeoutError("slow"), make_result("ok", 14, 3)]
        )

        with patch("summarizer.asyncio.sleep", new=AsyncMock()):
            output = await summarizer._call_with_retry(
                [{"role": "user", "content": "test"}],
                max_tokens=20,
                purpose="test retry",
            )

        self.assertEqual(output, "ok")
        self.assertEqual([record["status"] for record in summarizer.usage_records], ["failed", "success"])
        self.assertFalse(summarizer.usage_records[0]["usage"]["available"])
        self.assertEqual(summarizer.usage_records[1]["usage"]["total_tokens"], 17)

    async def test_effort_is_propagated_to_synthesis_request(self) -> None:
        summarizer = PaperSummarizer(
            api_key="test",
            base_url="https://example.com/v1",
            synthesis_streaming=False,
            synthesis_reasoning_effort="xhigh",
        )
        summarizer.client.achat_with_usage = AsyncMock(return_value=make_result("ok"))

        await summarizer._call_with_retry(
            [{"role": "user", "content": "test"}],
            max_tokens=20,
            purpose="test effort",
        )

        self.assertEqual(
            summarizer.client.achat_with_usage.await_args.kwargs["reasoning_effort"],
            "xhigh",
        )

    def test_invalid_effort_fails_before_client_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported synthesis reasoning effort"):
            PaperSummarizer(
                api_key="test",
                base_url="https://example.com/v1",
                synthesis_reasoning_effort="turbo",
            )

    def test_effort_environment_override_is_loaded(self) -> None:
        with patch.dict("os.environ", {"SYNTHESIS_REASONING_EFFORT": "low"}):
            config = Config.from_yaml("missing-config.yaml")
        self.assertEqual(config.synthesis_reasoning_effort, "low")


if __name__ == "__main__":
    unittest.main()
