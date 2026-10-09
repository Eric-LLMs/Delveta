"""``translate``: translate text into a target language via the shared LLM.

The target language is an OPTIONAL argument. When the sentence names one the
unified slot extractor resolves ``target_language`` and it flows here; when none
is named the executor applies a deterministic ENGLISH default (the default is the
executor's own constant, never a model output).
"""
from __future__ import annotations

from agent import Context, ToolExecution, ToolOutput, ToolRuntime, define_tool, text_block

# The confirmed default target when the request names none.
DEFAULT_TARGET_LANGUAGE = "English"


def register(runtime: ToolRuntime, ctx: Context, llm) -> None:
    async def translate(args: dict, exec: ToolExecution) -> str:
        # Bounded to a single line: the value rides into the system prompt, so a
        # stray newline from an upstream extraction cannot inject extra turns.
        target = ((args.get("target_language") or "").strip().splitlines() or [""])[0].strip()
        target = target or DEFAULT_TARGET_LANGUAGE
        return await llm.complete(
            args["text"],
            f"You are a translator. Translate the text into natural, fluent {target}.",
        )

    runtime.register(
        define_tool(
            name="translate",
            description="Translate text into a target language (default English). "
            "Pass target_language when the user names one; omit it to translate "
            "into English.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate."},
                    "target_language": {
                        "type": "string",
                        "description": "Target language name (e.g. 'English', "
                        "'Chinese'); omitted -> English.",
                    },
                },
                "required": ["text"],
            },
            output=ToolOutput(
                schema={"type": "string"}, render=lambda args, value: [text_block(value)]
            ),
            execute=translate,
        )
    )
