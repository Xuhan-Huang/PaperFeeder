"""
Universal LLM client supporting any OpenAI-compatible API.
Works with: OpenAI, Claude, Gemini, DeepSeek, Qwen, local models, etc.
"""

from __future__ import annotations

import base64
import httpx
from openai import OpenAI, AsyncOpenAI
from typing import Optional, Union, List
from pathlib import Path


class LLMClient:
    """
    Universal LLM client using OpenAI-compatible API format.
    
    Examples:
        # OpenAI
        client = LLMClient(
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="gpt-4o"
        )
        
        # Claude (via OpenAI-compatible endpoint)
        client = LLMClient(
            api_key="sk-ant-xxx",
            base_url="https://api.anthropic.com/v1",
            model="claude-sonnet-4-20250514"
        )
        
        # DeepSeek
        client = LLMClient(
            api_key="sk-xxx",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat"
        )
        
        # Gemini (via OpenAI-compatible endpoint)
        client = LLMClient(
            api_key="xxx",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.0-flash"
        )
        
        # Local (Ollama, vLLM, etc.)
        client = LLMClient(
            base_url="http://localhost:11434/v1",
            model="llama3"
        )
    """
    
    # 已知支持直接传 PDF 的模型前缀
    PDF_NATIVE_MODELS = ["claude", "gemini"]
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url
        
        # 检测是否是 Anthropic API（需要特殊处理）
        self.is_anthropic = "anthropic.com" in base_url
        
        if self.is_anthropic:
            # Anthropic 有自己的 SDK，但也可以用 OpenAI 兼容模式
            # 这里我们用原生 Anthropic SDK 以支持 PDF
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            self.client = OpenAI(
                api_key=api_key or "not-needed",  # 本地模型可能不需要
                base_url=base_url,
                timeout=httpx.Timeout(timeout),
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key or "not-needed",
                base_url=base_url,
                timeout=httpx.Timeout(timeout),
            )
    
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous chat completion."""
        if self.is_anthropic:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
    
    async def achat(
        self,
        messages: list[dict],
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        """Async chat completion."""
        if self.is_anthropic:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.content[0].text
        else:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
    
    def supports_pdf_native(self) -> bool:
        """Check if this model supports native PDF input."""
        model_lower = self.model.lower()
        return any(prefix in model_lower for prefix in self.PDF_NATIVE_MODELS)
    
    def chat_with_pdf(
        self,
        prompt: str,
        pdf_path: Optional[str] = None,
        pdf_url: Optional[str] = None,
        pdf_base64: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        Chat with a PDF document.
        
        Args:
            prompt: The user prompt
            pdf_path: Local path to PDF file
            pdf_url: URL to PDF (will be downloaded)
            pdf_base64: Base64-encoded PDF content
            max_tokens: Maximum response tokens
        """
        # Get PDF as base64
        if pdf_base64:
            pdf_data = pdf_base64
        elif pdf_path:
            pdf_data = self._file_to_base64(pdf_path)
        elif pdf_url:
            pdf_data = self._url_to_base64(pdf_url)
        else:
            raise ValueError("Must provide pdf_path, pdf_url, or pdf_base64")
        
        if self.is_anthropic:
            # Anthropic native PDF support
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.content[0].text
        
        elif self.supports_pdf_native():
            # Gemini-style (via OpenAI compat, may vary)
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "paper.pdf",
                            "file_data": f"data:application/pdf;base64,{pdf_data}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
            return self.chat(messages, max_tokens=max_tokens)
        
        else:
            # Fallback: extract text and send as text
            text = self._extract_pdf_text_from_base64(pdf_data)
            messages = [{
                "role": "user", 
                "content": f"{prompt}\n\n---\nPaper content:\n{text[:30000]}"
            }]
            return self.chat(messages, max_tokens=max_tokens)
    
    def _file_to_base64(self, path: str) -> str:
        """Read file and convert to base64."""
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    def _url_to_base64(self, url: str) -> str:
        """Download URL and convert to base64."""
        import httpx
        response = httpx.get(url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        return base64.standard_b64encode(response.content).decode("utf-8")
    
    def _extract_pdf_text_from_base64(self, pdf_base64: str) -> str:
        """Extract text from base64-encoded PDF."""
        try:
            import fitz  # PyMuPDF
            pdf_bytes = base64.b64decode(pdf_base64)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            return "[PDF text extraction unavailable - install pymupdf]"
        except Exception as e:
            return f"[PDF extraction error: {e}]"


# Convenience presets
def openai_client(api_key: str, model: str = "gpt-4o-mini") -> LLMClient:
    return LLMClient(api_key=api_key, base_url="https://api.openai.com/v1", model=model)

def claude_client(api_key: str, model: str = "claude-sonnet-4-20250514") -> LLMClient:
    return LLMClient(api_key=api_key, base_url="https://api.anthropic.com/v1", model=model)

def deepseek_client(api_key: str, model: str = "deepseek-chat") -> LLMClient:
    return LLMClient(api_key=api_key, base_url="https://api.deepseek.com/v1", model=model)

def gemini_client(api_key: str, model: str = "gemini-2.0-flash") -> LLMClient:
    return LLMClient(
        api_key=api_key, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model=model
    )

def qwen_client(api_key: str, model: str = "qwen-turbo") -> LLMClient:
    return LLMClient(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model
    )

def local_client(base_url: str = "http://localhost:11434/v1", model: str = "llama3") -> LLMClient:
    return LLMClient(base_url=base_url, model=model)
