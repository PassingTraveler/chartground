from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MMTemplate:
    image_token: str = "<|image_pad|>"
    video_token: str = "<|video_pad|>"
    audio_start: str = "<|audio_start|>"
    audio_end: str = "<|audio_end|>"
    audio_pad: str = "<|audio_pad|>"
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    assistant: str = "assistant"
    user: str = "user"

    def image_markers(self, n_images: int = 1, tokens_per_image: int = 64) -> str:
        return self.image_token * (n_images * tokens_per_image)

    def video_markers(self, n_frames: int, tokens_per_frame: int = 64) -> str:
        return self.video_token * (n_frames * tokens_per_frame)

    def prompt(self, question: str, modality: str = "text", n_images: int = 0, n_frames: int = 0,
               tokens_per_image: int = 64, tokens_per_frame: int = 64) -> str:
        if modality == "image":
            visual = self.image_markers(n_images or 1, tokens_per_image)
            body = f"{visual}\n{question}"
        elif modality == "video":
            visual = self.video_markers(n_frames or 1, tokens_per_frame)
            body = f"{visual}\n{question}"
        else:
            body = question
        return f"{self.im_start}{self.user}\n{body}{self.im_end}\n{self.im_start}{self.assistant}\n"

    def prompt_audio(self, question: str, n_images: int = 1, tokens_per_image: int = 64,
                     n_audio_tokens: int = 256, include_question: bool = False) -> str:
        """Audio-question prompt: image markers first, then the audio span.

        The question text is omitted by default so the model must actually
        parse the audio (a text arm would make Delta_audio unmeasurable);
        `include_question` supports a diagnostic/curriculum arm.
        """
        visual = self.image_markers(n_images or 1, tokens_per_image)
        audio = f"{self.audio_start}{self.audio_pad * n_audio_tokens}{self.audio_end}"
        body = f"{visual}\n{audio}" + (f"\n{question}" if include_question else "")
        return f"{self.im_start}{self.user}\n{body}{self.im_end}\n{self.im_start}{self.assistant}\n"

    def target(self, answer: str) -> str:
        return answer + self.im_end

