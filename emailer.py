"""
Email sending module.
Supports Resend and SendGrid.
"""

import aiohttp
import re
from abc import ABC, abstractmethod
from html import unescape
from html.parser import HTMLParser
from typing import Optional


_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>[\s\S]*?</script\s*>", re.IGNORECASE)
_SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*?/?>", re.IGNORECASE)


def sanitize_email_html(html_content: str) -> str:
    """Remove executable content that should never be present in email HTML."""
    without_script_blocks = _SCRIPT_BLOCK_RE.sub("", html_content or "")
    return _SCRIPT_TAG_RE.sub("", without_script_blocks)


class _EmailTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
        "table",
        "tr",
    }
    _SUPPRESSED_TAGS = {"head", "script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed_depth = 0
        self.link_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self._SUPPRESSED_TAGS:
            self.suppressed_depth += 1
            return
        if self.suppressed_depth:
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a":
            href = next((value or "" for name, value in attrs if name.lower() == "href"), "")
            self.link_stack.append(href.strip())

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SUPPRESSED_TAGS:
            self.suppressed_depth = max(0, self.suppressed_depth - 1)
            return
        if self.suppressed_depth:
            return
        if tag == "a" and self.link_stack:
            href = self.link_stack.pop()
            if href:
                self.parts.append(f" ({href})")
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.suppressed_depth:
            self.parts.append(data)


def html_to_plain_text(html_content: str) -> str:
    """Create a readable plain-text alternative while preserving link targets."""
    parser = _EmailTextExtractor()
    parser.feed(html_content or "")
    parser.close()
    lines = [" ".join(unescape(line).split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line).strip()


class BaseEmailer(ABC):
    """Abstract base class for email services."""
    
    @abstractmethod
    async def send(
        self,
        to: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> bool:
        pass


class ResendEmailer(BaseEmailer):
    """Send emails using Resend API."""
    
    API_URL = "https://api.resend.com/emails"
    
    def __init__(self, api_key: str, from_email: str = "paperfeeder@resend.dev"):
        self.api_key = api_key
        self.from_email = from_email
    
    async def send(
        self,
        to: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> bool:
        """Send an email using Resend."""
        safe_html_content = sanitize_email_html(html_content)
        safe_text_content = text_content or html_to_plain_text(safe_html_content)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "from": self.from_email,
            "to": [to] if isinstance(to, str) else to,
            "subject": subject,
            "html": safe_html_content,
        }
        
        if safe_text_content:
            payload["text"] = safe_text_content
        if attachments:
            payload["attachments"] = attachments
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.API_URL,
                headers=headers,
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    email_id = str(result.get("id", "")).strip()
                    if email_id:
                        print(f"   Resend email ID: {email_id}")
                    return True
                else:
                    error = await response.text()
                    print(f"Resend error: {response.status} - {error}")
                    return False


class SendGridEmailer(BaseEmailer):
    """Send emails using SendGrid API."""
    
    API_URL = "https://api.sendgrid.com/v3/mail/send"
    
    def __init__(self, api_key: str, from_email: str):
        self.api_key = api_key
        self.from_email = from_email
    
    async def send(
        self,
        to: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> bool:
        """Send an email using SendGrid."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "personalizations": [{"to": [{"email": to}]}],
            "from": {"email": self.from_email},
            "subject": subject,
            "content": [{"type": "text/html", "value": html_content}],
        }
        
        if text_content:
            payload["content"].insert(0, {"type": "text/plain", "value": text_content})
        if attachments:
            payload["attachments"] = [
                {
                    "content": a.get("content", ""),
                    "filename": a.get("filename", "attachment.bin"),
                    "type": a.get("content_type", "application/octet-stream"),
                    "disposition": "attachment",
                }
                for a in attachments
            ]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.API_URL,
                headers=headers,
                json=payload,
            ) as response:
                if response.status in (200, 202):
                    return True
                else:
                    error = await response.text()
                    print(f"SendGrid error: {response.status} - {error}")
                    return False


class ConsoleEmailer(BaseEmailer):
    """Mock emailer that prints to console (for testing)."""
    
    async def send(
        self,
        to: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> bool:
        """Print email to console."""
        print("\n" + "=" * 60)
        print(f"TO: {to}")
        print(f"SUBJECT: {subject}")
        print("=" * 60)
        print(html_content[:2000])
        if len(html_content) > 2000:
            print(f"\n... [{len(html_content) - 2000} more characters]")
        print("=" * 60 + "\n")
        return True


class FileEmailer(BaseEmailer):
    """Save email to file (for testing/preview)."""
    
    def __init__(self, output_path: str = "email_preview.html"):
        self.output_path = output_path
    
    async def send(
        self,
        to: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        attachments: Optional[list[dict]] = None,
    ) -> bool:
        """Save email to file."""
        try:
            with open(self.output_path, "w") as f:
                f.write(f"<!-- TO: {to} -->\n")
                f.write(f"<!-- SUBJECT: {subject} -->\n")
                f.write(html_content)
            print(f"Email saved to {self.output_path}")
            return True
        except Exception as e:
            print(f"Error saving email: {e}")
            return False
