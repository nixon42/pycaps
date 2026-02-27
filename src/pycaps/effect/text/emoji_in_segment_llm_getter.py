from typing import Optional
from pycaps.common import Document, Segment
from pycaps.ai import LlmProvider
from pycaps.utils import ScriptUtils
from pycaps.logger import logger


class EmojiInSegmentLlmGetter:
    def __init__(self) -> None:
        self._llm = None
        self._summary = ""

    def start(self, document: Document) -> None:
        self._llm = LlmProvider.get()
        self._summary = ScriptUtils.get_basic_summary(document.get_text())

    def get_emoji(self, segment: Segment) -> Optional[str]:
        text = segment.get_text()
        prompt = f"""
            Given the following subtitle text, decide whether it meaningfully conveys an emotion, action, or idea that can be represented with an emoji.
            If it does you will need to respond with a single, appropriate emoji only.

            Take into account that the subtitle is part of a video script. This is a video script summary:
            {self._summary}

            Basic guidelines:
            1. The emoji should be related to subtitle text received.
            2. Use the video script summary to get better context about the meaning of the words in the subtitle.
            3. Respond only with the emoji, no other text.
            4. If the text received doesn't contain any relevant information (e.g. it's too vague, neutral, or generic), respond with "None".

            STRICT OUTPUT MODE:
            - No Markdown.
            - No explanations.
            - No extra commentary.
            - just emoji or "None"

            Subtitle to analyze: "{text}"
            """
        prompt = f"""
Task: Decide if the subtitle expresses a clear emotion, action, or idea that can be represented with ONE emoji.

Context:
Video summary: {self._summary}

Rules:
- Use the summary for context.
- If suitable, return exactly ONE relevant emoji.
- If vague/neutral/generic, return: None
- Output ONLY the emoji or None.
- No text, no explanation, no markdown.

Subtitle: "{text}"
        """
        text_response = self._llm.send_message(prompt=prompt)
        logger().info(
            f"[EmojiInSegmentLlmGetter] Emoji for segment '{text}': {text_response}"
        )
        if text_response == "None":
            return None

        return text_response
