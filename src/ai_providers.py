"""
PHAT AI Provider Abstraction Layer
Supports Claude, Gemini, and Deepseek APIs with unified interface

Enhanced with custom model selection support for per-task model assignment.
"""

import json
import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enumeration of AI task types for custom model selection"""
    TRIAGE = "triage"
    CLAIMS_EXTRACTION = "claims_extraction"
    REJECTION_ANALYSIS = "rejection_analysis"
    AMENDMENT_ANALYSIS = "amendment_analysis"
    ARGUMENT_EXTRACTION = "argument_extraction"
    RESTRICTION_ANALYSIS = "restriction_analysis"
    ALLOWANCE_ANALYSIS = "allowance_analysis"
    INTERVIEW_ANALYSIS = "interview_analysis"
    TERMINAL_DISCLAIMER = "terminal_disclaimer"
    MEANS_PLUS_FUNCTION = "means_plus_function"
    DOCUMENT_CLASSIFICATION = "document_classification"
    GENERIC_ANALYSIS = "generic_analysis"
    # Post-processing (Layer 2/3 synthesis tasks)
    SHADOW_EXAMINER = "shadow_examiner"
    DEFINITION_SYNTHESIS = "definition_synthesis"
    CLAIM_NARRATIVE = "claim_narrative"
    STRATEGIC_TENSIONS = "strategic_tensions"
    THEMATIC_SYNTHESIS = "thematic_synthesis"
    TERM_BOUNDARIES = "term_boundaries"
    VULNERABILITY_CARDS = "vulnerability_cards"


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    # Class-level verbose flag (set by factory from config)
    _verbose = False
    
    def __init__(self, api_key: str, model: str, temperature: float = 0.1):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 2
    
    @classmethod
    def set_verbose(cls, verbose: bool):
        """Enable or disable verbose logging for all providers"""
        cls._verbose = verbose
        if verbose:
            logger.info("VERBOSE MODE ENABLED - All AI inputs/outputs will be logged")
    
    def _log_verbose(self, message: str, data: str = None):
        """Log verbose message if verbose mode is enabled"""
        if self._verbose:
            logger.info(f"[VERBOSE] {message}")
            if data:
                # Truncate very long data but show enough to be useful
                if len(data) > 5000:
                    logger.info(f"[VERBOSE] Data (truncated {len(data)} chars):\n{data[:2500]}\n...[TRUNCATED]...\n{data[-2500:]}")
                else:
                    logger.info(f"[VERBOSE] Data:\n{data}")
    
    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a completion for the given prompt"""
        pass
    
    @abstractmethod
    def complete_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate a JSON completion for the given prompt"""
        pass
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff.
        
        Uses longer delays for 429 rate limit errors since these require
        waiting for the rate limit window to reset.
        Does NOT retry 400 errors (client errors that won't fix themselves).
        """
        last_exception = None
        max_attempts = self.max_retries + 2  # Extra retries for rate limits
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # 400 Bad Request: non-retriable client error (wrong params,
                # context length exceeded, etc.). Retrying won't help.
                if '400' in error_str and 'Bad Request' in error_str:
                    logger.error(f"Non-retriable 400 error, failing immediately: {e}")
                    raise
                
                # Detect 429 rate limit errors and use longer backoff
                is_rate_limit = '429' in error_str
                if is_rate_limit:
                    # Rate limit: start at 15s, then 30s, 60s, 60s, 60s
                    wait_time = min(60, 15 * (2 ** attempt))
                else:
                    wait_time = self.retry_delay * (2 ** attempt)
                    if attempt >= self.max_retries - 1:
                        raise last_exception
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        raise last_exception
    
    def _extract_and_parse_json(self, response: str) -> Dict[str, Any]:
        """
        Robustly extract and parse JSON from an AI response.
        Handles markdown blocks, truncation, and common formatting issues.
        """
        original_response = response
        response = response.strip()
        
        # Step 1: Remove markdown code blocks
        if "```json" in response:
            # Find content between ```json and ```
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
            else:
                response = response[start:].strip()
        elif "```" in response:
            # Generic code block
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
            else:
                response = response[start:].strip()
        
        # Step 2: Find JSON boundaries (look for outermost { } or [ ])
        json_start = -1
        json_end = -1
        
        for i, char in enumerate(response):
            if char in '{[':
                json_start = i
                break
        
        if json_start != -1:
            # Find matching closing bracket
            open_char = response[json_start]
            close_char = '}' if open_char == '{' else ']'
            depth = 0
            in_string = False
            escape_next = False
            
            for i in range(json_start, len(response)):
                char = response[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == open_char:
                        depth += 1
                    elif char == close_char:
                        depth -= 1
                        if depth == 0:
                            json_end = i + 1
                            break
            
            if json_end != -1:
                response = response[json_start:json_end]
        
        # Step 3: Try to parse
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            
            # Step 4: Try to fix common issues
            fixed_response = self._attempt_json_repair(response)
            if fixed_response:
                try:
                    return json.loads(fixed_response)
                except json.JSONDecodeError:
                    pass
            
            # Step 5: Log the problematic response for debugging
            logger.error(f"Failed to parse JSON response. First 500 chars: {original_response[:500]}")
            logger.error(f"Last 500 chars: {original_response[-500:]}")
            
            raise
    
    def _attempt_json_repair(self, response: str) -> Optional[str]:
        """Attempt to repair common JSON issues like truncation"""
        
        # Track the opening brackets/braces in order
        stack = []
        in_string = False
        escape_next = False
        
        for char in response:
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char in '}]':
                    if stack and stack[-1] == char:
                        stack.pop()
        
        # If we ended inside a string, close it
        if in_string:
            # Bug Fix C (v2.2): If the last string value ends abruptly (no
            # sentence-ending punctuation), append "..." before closing the
            # quotes so the user can see the text was truncated by the model.
            stripped = response.rstrip()
            if stripped and stripped[-1] not in '.!?"\'':
                response = response + '..."'
            else:
                response = response + '"'
        
        # Close any open structures in reverse order (LIFO)
        if stack:
            logger.warning(
                f"JSON REPAIR: Closing {len(stack)} unclosed bracket(s). "
                f"This typically means the AI response was truncated by the token limit. "
                f"Data beyond the truncation point has been LOST."
            )
            response = response + ''.join(reversed(stack))
            return response
        
        # Fix 2: Control characters in strings
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', response)
        if cleaned != response:
            return cleaned
        
        return None


class ClaudeProvider(AIProvider):
    """Anthropic Claude API provider"""
    
    API_URL = "https://api.anthropic.com/v1/messages"
    
    # Max output tokens per model (from Anthropic docs)
    # Claude 4.x Sonnet/Haiku 4.5: 64K output
    # Claude 4 Opus: 32K output
    # Claude 3.x Haiku 3.5: 8K output
    MAX_OUTPUT_TOKENS = {
        "claude-sonnet-4-20250514": 64000,
        "claude-sonnet-4-5-20250929": 64000,
        "claude-sonnet-4-5": 64000,
        "claude-sonnet-4-6": 64000,
        "claude-opus-4-20250514": 32000,
        "claude-opus-4-5": 64000,
        "claude-opus-4-6": 64000,
        "claude-haiku-4-5-20251001": 64000,
        "claude-haiku-4-5": 64000,
        "claude-3-5-haiku-20241022": 8192,
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-7-sonnet-20250219": 64000,
    }
    DEFAULT_MAX_OUTPUT = 64000
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5-20250929", temperature: float = 0.1):
        super().__init__(api_key, model, temperature)
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        self._max_tokens = self.MAX_OUTPUT_TOKENS.get(model, self.DEFAULT_MAX_OUTPUT)
        logger.info(f"Claude provider: model={model}, max_output_tokens={self._max_tokens}")
    
    def _make_request(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a request to the Claude API"""
        self._log_verbose(f"CLAUDE API REQUEST to {self.model}")
        
        data = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        with httpx.Client(timeout=180.0) as client:
            response = client.post(self.API_URL, headers=self.headers, json=data)
            
            # Log error response body before raising
            if response.status_code != 200:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text[:500]
                logger.error(f"CLAUDE API ERROR {response.status_code}: {error_body}")
            
            response.raise_for_status()
            result = response.json()
        
        # Detect truncation due to token limit
        stop_reason = result.get("stop_reason", "")
        if stop_reason == "max_tokens":
            logger.warning(
                f"CLAUDE RESPONSE TRUNCATED: hit max_tokens limit. "
                f"Response may be incomplete."
            )
        
        response_text = result["content"][0]["text"]
        self._log_verbose("CLAUDE API RESPONSE:", response_text)
        
        return response_text
    
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a completion for the given prompt"""
        return self._retry_with_backoff(self._make_request, prompt, system_prompt)
    
    def complete_json(self, prompt: str, system_prompt: Optional[str] = None, max_json_retries: int = 2) -> Dict[str, Any]:
        """Generate a JSON completion with robust parsing and retry on failure"""
        json_system = (system_prompt or "") + """

CRITICAL: Respond with valid JSON only.
- No markdown code blocks (no ```)
- No explanatory text before or after
- Ensure all strings are properly escaped
- Ensure the JSON is complete (all brackets closed)"""

        last_error = None
        for attempt in range(max_json_retries + 1):
            try:
                response = self.complete(prompt, json_system)
                parsed = self._extract_and_parse_json(response)
                self._log_verbose(f"JSON PARSED SUCCESSFULLY (attempt {attempt + 1})", json.dumps(parsed, indent=2)[:2000])
                return parsed
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_json_retries:
                    logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}, retrying...")
        
        raise last_error


class GeminiProvider(AIProvider):
    """Google Gemini API provider"""
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    # Max output tokens per model (from Google docs)
    # Gemini 2.5 series: 65536 output tokens
    # Gemini 1.5 Pro: 8192 output tokens
    # Gemini 1.5 Flash: 8192 output tokens
    MAX_OUTPUT_TOKENS = {
        "gemini-2.5-pro": 65536,
        "gemini-2.5-flash": 65536,
        "gemini-2.5-flash-lite": 65536,
        "gemini-2.0-flash": 8192,
        "gemini-1.5-pro": 8192,
        "gemini-1.5-flash": 8192,
    }
    DEFAULT_MAX_OUTPUT = 65536
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro", temperature: float = 0.1):
        super().__init__(api_key, model, temperature)
        self._max_tokens = self.MAX_OUTPUT_TOKENS.get(model, self.DEFAULT_MAX_OUTPUT)
        logger.info(f"Gemini provider: model={model}, max_output_tokens={self._max_tokens}")
    
    def _make_request(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a request to the Gemini API"""
        self._log_verbose(f"GEMINI API REQUEST to {self.model}")
        
        url = self.API_URL.format(model=self.model) + f"?key={self.api_key}"
        
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System Instructions: {system_prompt}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "I understand and will follow these instructions."}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self._max_tokens,
            }
        }
        
        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, json=data)
            
            # Log error response body before raising
            if response.status_code != 200:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text[:500]
                logger.error(f"GEMINI API ERROR {response.status_code}: {error_body}")
            
            response.raise_for_status()
            result = response.json()
        
        # Detect truncation due to token limit
        finish_reason = result.get("candidates", [{}])[0].get("finishReason", "")
        if finish_reason == "MAX_TOKENS":
            logger.warning(
                f"GEMINI RESPONSE TRUNCATED: hit maxOutputTokens limit. "
                f"Response may be incomplete."
            )
        
        response_text = result["candidates"][0]["content"]["parts"][0]["text"]
        self._log_verbose("GEMINI API RESPONSE:", response_text)
        
        return response_text
    
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a completion for the given prompt"""
        return self._retry_with_backoff(self._make_request, prompt, system_prompt)
    
    def complete_json(self, prompt: str, system_prompt: Optional[str] = None, max_json_retries: int = 2) -> Dict[str, Any]:
        """Generate a JSON completion with robust parsing and retry on failure"""
        json_system = (system_prompt or "") + """

CRITICAL: Respond with valid JSON only.
- No markdown code blocks (no ```)
- No explanatory text before or after
- Ensure all strings are properly escaped
- Ensure the JSON is complete (all brackets closed)"""

        last_error = None
        for attempt in range(max_json_retries + 1):
            try:
                response = self.complete(prompt, json_system)
                parsed = self._extract_and_parse_json(response)
                self._log_verbose(f"JSON PARSED SUCCESSFULLY (attempt {attempt + 1})", json.dumps(parsed, indent=2)[:2000])
                return parsed
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_json_retries:
                    logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}, retrying...")
        
        raise last_error


class DeepseekProvider(AIProvider):
    """Deepseek API provider (OpenAI-compatible)"""
    
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    # Max output tokens per model (from DeepSeek docs, Feb 2025)
    # deepseek-reasoner (V3.2 Thinking Mode): default 32K, max 64K output tokens
    # deepseek-chat (V3.2 Non-thinking Mode): default 4K, max 8K output tokens
    MAX_OUTPUT_TOKENS = {
        "deepseek-reasoner": 64000,
        "deepseek-chat": 8192,
    }
    DEFAULT_MAX_OUTPUT = 8192
    
    def __init__(self, api_key: str, model: str = "deepseek-reasoner", temperature: float = 0.1):
        super().__init__(api_key, model, temperature)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._max_tokens = self.MAX_OUTPUT_TOKENS.get(model, self.DEFAULT_MAX_OUTPUT)
        logger.info(f"DeepSeek provider: model={model}, max_output_tokens={self._max_tokens}")
    
    def _make_request(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a request to the Deepseek API"""
        self._log_verbose(f"DEEPSEEK API REQUEST to {self.model}")
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self._max_tokens
        }
        
        with httpx.Client(timeout=180.0) as client:
            response = client.post(self.API_URL, headers=self.headers, json=data)
            
            # Log error response body before raising (critical for diagnosing 400s)
            if response.status_code != 200:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text[:500]
                logger.error(f"DEEPSEEK API ERROR {response.status_code}: {error_body}")
            
            response.raise_for_status()
            result = response.json()
        
        # Detect truncation due to token limit
        finish_reason = result.get("choices", [{}])[0].get("finish_reason", "")
        if finish_reason == "length":
            logger.warning(
                f"DEEPSEEK RESPONSE TRUNCATED: hit max_tokens limit. "
                f"Response may be incomplete."
            )
        
        response_text = result["choices"][0]["message"]["content"]
        self._log_verbose("DEEPSEEK API RESPONSE:", response_text)
        
        return response_text
    
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a completion for the given prompt"""
        return self._retry_with_backoff(self._make_request, prompt, system_prompt)
    
    def complete_json(self, prompt: str, system_prompt: Optional[str] = None, max_json_retries: int = 2) -> Dict[str, Any]:
        """Generate a JSON completion with robust parsing and retry on failure"""
        json_system = (system_prompt or "") + """

CRITICAL: Respond with valid JSON only.
- No markdown code blocks (no ```)
- No explanatory text before or after
- Ensure all strings are properly escaped
- Ensure the JSON is complete (all brackets closed)"""

        last_error = None
        for attempt in range(max_json_retries + 1):
            try:
                response = self.complete(prompt, json_system)
                parsed = self._extract_and_parse_json(response)
                self._log_verbose(f"JSON PARSED SUCCESSFULLY (attempt {attempt + 1})", json.dumps(parsed, indent=2)[:2000])
                return parsed
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_json_retries:
                    logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}, retrying...")
        
        raise last_error


class ModelSelector:
    """
    Manages model selection for different task types.
    
    Supports three modes:
    - "full": Always use the default (best) model
    - "fast": Always use the fast (cheaper) model
    - "custom": Use per-task model assignments from config
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model selector.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.provider_name = config.get("ai_provider", "claude")
        self.model_mode = config.get("model_mode", "full").lower()
        
        # Get model names for this provider
        provider_models = config.get("models", {}).get(self.provider_name, {})
        self.full_model = provider_models.get("default", "")
        self.fast_model = provider_models.get("fast", self.full_model)
        
        # Get custom assignments if in custom mode
        self.custom_assignments = config.get("custom_model_assignments", {})
        
        logger.info(f"ModelSelector initialized: mode={self.model_mode}, "
                   f"provider={self.provider_name}, full={self.full_model}, fast={self.fast_model}")
    
    def get_model_for_task(self, task: TaskType) -> str:
        """
        Get the appropriate model for a given task.
        
        Args:
            task: The TaskType enum value
            
        Returns:
            Model name string
        """
        if self.model_mode == "fast":
            return self.fast_model
        elif self.model_mode == "custom":
            # Look up the assignment for this task
            assignment = self.custom_assignments.get(task.value, "full")
            model = self.fast_model if assignment == "fast" else self.full_model
            logger.debug(f"Custom model selection: {task.value} -> {assignment} -> {model}")
            return model
        else:  # "full" or default
            return self.full_model
    
    def get_tier_for_task(self, task: TaskType) -> str:
        """
        Get the tier (full/fast) for a given task.
        
        Args:
            task: The TaskType enum value
            
        Returns:
            "full" or "fast"
        """
        if self.model_mode == "fast":
            return "fast"
        elif self.model_mode == "custom":
            return self.custom_assignments.get(task.value, "full")
        else:
            return "full"
    
    def get_summary(self) -> Dict[str, str]:
        """Get a summary of model assignments for all tasks"""
        summary = {"mode": self.model_mode}
        for task in TaskType:
            tier = self.get_tier_for_task(task)
            model = self.get_model_for_task(task)
            summary[task.value] = f"{tier} ({model})"
        return summary


class AIProviderFactory:
    """Factory for creating AI providers based on configuration"""
    
    PROVIDERS = {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "deepseek": DeepseekProvider
    }
    
    @classmethod
    def create(cls, provider_name: str, api_key: str, model: str, temperature: float = 0.1, verbose: bool = False) -> AIProvider:
        """Create an AI provider instance"""
        provider_name = provider_name.lower()
        
        if provider_name not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(cls.PROVIDERS.keys())}")
        
        # Set verbose mode on the base class
        AIProvider.set_verbose(verbose)
        
        provider_class = cls.PROVIDERS[provider_name]
        return provider_class(api_key=api_key, model=model, temperature=temperature)
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> AIProvider:
        """Create an AI provider from a configuration dictionary.
        
        Respects 'model_mode' setting: 'full' (default), 'fast', or 'custom'
        For 'custom' mode, this returns a provider with the full model;
        use create_for_task() for task-specific model selection.
        """
        provider_name = config.get("ai_provider", "claude")
        api_key = config["api_keys"][provider_name]
        
        # Select model based on model_mode setting
        model_mode = config.get("model_mode", "full").lower()
        if model_mode == "fast":
            model = config["models"][provider_name].get("fast", config["models"][provider_name]["default"])
        else:
            # For both "full" and "custom" modes, default to full model
            model = config["models"][provider_name]["default"]
        
        temperature = config.get("processing", {}).get("temperature", 0.1)
        
        # Check verbose setting
        verbose = config.get("logging", {}).get("verbose", False)
        
        return cls.create(provider_name, api_key, model, temperature, verbose)
    
    @classmethod
    def create_fast_from_config(cls, config: Dict[str, Any]) -> AIProvider:
        """Create a fast AI provider (always uses fast model, ignores model_mode)"""
        provider_name = config.get("ai_provider", "claude")
        api_key = config["api_keys"][provider_name]
        model = config["models"][provider_name].get("fast", config["models"][provider_name]["default"])
        temperature = config.get("processing", {}).get("temperature", 0.1)
        
        # Check verbose setting
        verbose = config.get("logging", {}).get("verbose", False)
        
        return cls.create(provider_name, api_key, model, temperature, verbose)
    
    @classmethod
    def create_for_task(cls, config: Dict[str, Any], task: TaskType) -> AIProvider:
        """
        Create an AI provider for a specific task.
        
        This respects custom model assignments when model_mode is "custom".
        
        Args:
            config: Configuration dictionary
            task: The task type for model selection
            
        Returns:
            AIProvider configured for the task
        """
        selector = ModelSelector(config)
        model = selector.get_model_for_task(task)
        
        provider_name = config.get("ai_provider", "claude")
        api_key = config["api_keys"][provider_name]
        temperature = config.get("processing", {}).get("temperature", 0.1)
        verbose = config.get("logging", {}).get("verbose", False)
        
        logger.debug(f"Creating provider for task {task.value}: {provider_name}/{model}")
        
        return cls.create(provider_name, api_key, model, temperature, verbose)
    
    @classmethod
    def get_model_info(cls, config: Dict[str, Any]) -> Dict[str, str]:
        """Return info about which provider/model will be used"""
        provider_name = config.get("ai_provider", "claude")
        model_mode = config.get("model_mode", "full").lower()
        
        if model_mode == "fast":
            model = config["models"][provider_name].get("fast", config["models"][provider_name]["default"])
        else:
            model = config["models"][provider_name]["default"]
        
        return {
            "provider": provider_name,
            "model": model,
            "mode": model_mode
        }
    
    @classmethod
    def set_verbose_from_config(cls, config: Dict[str, Any]):
        """Set verbose mode from config (call this if provider already created)"""
        verbose = config.get("logging", {}).get("verbose", False)
        AIProvider.set_verbose(verbose)


if __name__ == "__main__":
    # Test the providers and model selector
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Test model selector
    selector = ModelSelector(config)
    print("Model Selection Summary:")
    for task, assignment in selector.get_summary().items():
        print(f"  {task}: {assignment}")
    
    # Test provider creation
    provider = AIProviderFactory.create_from_config(config)
    response = provider.complete("What is 2 + 2? Answer briefly.")
    print(f"\nResponse: {response}")
