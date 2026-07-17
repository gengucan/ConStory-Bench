# Add delays in the script

"""
ConStory-Bench custom generation pipeline using Mem0 agentic memory.

Research question: To what extent can a Mem0-based agentic memory layer enable
non-reasoning models to achieve narrative consistency comparable to reasoning
models in 8000-10000 word fiction, as measured by ConStory-Bench?

Experimental design:
  Baseline   — reasoning model (e.g. o3-mini), one-shot, full context
  Experimental — non-reasoning model (e.g. claude-haiku-4-5) + Mem0, chunked,
                 full accumulated story in context each turn

Usage:
    python generate.py \
        --prompts data/prompts/constory_prompts.parquet \
        --model claude-haiku-4-5-20251001 \
        --output data/stories/claude-haiku-4-5_mem0.parquet

Then pass the output to ConStory's judge:
    python -m constory.judge \
        --input data/stories/claude-haiku-4-5_mem0.parquet \
        --story-column generated_story \
        --model-name claude-haiku-4-5_mem0 \
        --concurrent 3
"""

import os
import uuid
import argparse
import pandas as pd
import time
from datetime import datetime, timezone
import anthropic
from mem0 import MemoryClient

# ── Clients ───────────────────────────────────────────────────────────────────

anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
mem0_client       = MemoryClient(api_key=os.environ["MEM0_API_KEY"])

# ── Config ────────────────────────────────────────────────────────────────────

CHUNK_WORDS = 1_000
STORY_MIN   = 8_000
STORY_MAX   = 10_000
MAX_TOKENS  = 4_096

SYSTEM_INSTRUCTIONS = """
You are a fiction writer working on a long-form story in chunks.

If you are starting a story:
- Take the prompt and establish character, setting, and plot.
- Write approximately {chunk_words} words then stop mid-story (do not resolve the plot).

If you are continuing a story:
- You will be given the full story written so far plus memory notes retrieved from Mem0.
- Continue seamlessly from where the story ends, preserving tone, style, and all established facts.
- Write approximately {chunk_words} more words then stop again at a natural pause point.

MEMORY HANDLING (internal only):
- Every 1000 words, call the Mem0 tool to store structured notes covering:
  - Major character details (names, traits, relationships, arcs)
  - World rules and setting facts
  - Key plot points that have occurred
  - Your intended direction for the plot going forward
  - Tone and stylistic choices
  - Current total word count (as a separate memory entry)
- This is a tool call, not part of your written response. Never paraphrase, summarize, or reference these notes in the story output.

OUTPUT FORMAT:
- Your response must contain ONLY the story prose for this chunk.
- Do not include memory notes, summaries, meta-commentary, word counts, chunk labels, headers, or any text about what you stored or plan to store.
- Do not wrap the story in quotes, titles, or preambles like "Here's the next part" — start directly with story text and end directly with story text.
""".strip()


# ── Core generation functions ─────────────────────────────────────────────────

def _system_prompt(mem0_context: str) -> str:
    instructions = SYSTEM_INSTRUCTIONS.format(chunk_words=CHUNK_WORDS)
    if mem0_context:
        return f"{instructions}\n\n--- Extracted memory notes ---\n{mem0_context}"
    return instructions


def _retrieve_context(query: str, user_id: str) -> str:
    results = mem0_client.search(query, filters={"user_id": user_id}, top_k=10)
    return "\n".join(m["memory"] for m in results["results"])


def _generate_chunk(
    user_message: str,
    user_id: str,
    model: str,
    story_so_far: str,
) -> str:
    """
    Call the LLM with:
      - System prompt containing Mem0 memory notes
      - Full accumulated story so far (mirrors reasoning model's full context)
      - Continuation instruction as the final user turn
    """
    mem0_context = _retrieve_context(user_message, user_id)

    messages = []

    # Pass the full story so far so the model has identical prose context to
    # what a reasoning model would have in a one-shot call.
    if story_so_far:
        messages.append({
            "role": "user",
            "content": (
                "Here is the full story written so far. "
                "Continue it in your next response — do not repeat any of it:\n\n"
                f"{story_so_far}"
            ),
        })
        messages.append({
            "role": "assistant",
            "content": (
                "Understood. I have the full story and my memory notes. "
                "I will continue seamlessly from where it ends."
            ),
        })

    messages.append({"role": "user", "content": user_message})

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(mem0_context),
        messages=messages,
    ).content[0].text

    # Store exchange so Mem0 can extract and consolidate structured memories
    mem0_client.add(
        [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": response},
        ],
        user_id=user_id,
    )

    return response


def generate_story(prompt: str, model: str, story_id) -> str:
    """
    Iteratively generates ~CHUNK_WORDS chunks until STORY_MIN words are
    accumulated, passing the full story and Mem0 memories into each turn.
    """
    # Unique isolated user_id per story — prevents memory bleed between
    # stories and between model conditions.
    user_id = f"{model}__{story_id}__{uuid.uuid4().hex[:8]}"

    chunks     = []
    word_count = 0

    # ── First chunk: establish the story ─────────────────────────────────────
    first_message = (
        f"{prompt}\n\n"
        f"Write only the opening ~{CHUNK_WORDS} words of the story. "
        f"Establish character, setting, and plot — but do not resolve anything. "
        f"Stop at a natural pause point."
    )
    chunk = _generate_chunk(first_message, user_id, model, story_so_far="")
    chunks.append(chunk)
    word_count += len(chunk.split())
    print(f"  chunk 1: {word_count} words so far")

    # ── Continuation chunks ───────────────────────────────────────────────────
    chunk_num = 2
    while word_count < STORY_MIN:
        time.sleep(1)
        remaining    = STORY_MAX - word_count
        story_so_far = "\n\n".join(chunks)

        continuation = (
            f"Continue the story. Write approximately "
            f"{min(CHUNK_WORDS, remaining)} more words. "
            f"Current total word count is {word_count}. "
            f"Do not resolve the plot unless you are within 500 words of {STORY_MAX}."
        )
        chunk = _generate_chunk(continuation, user_id, model, story_so_far)
        chunks.append(chunk)
        word_count += len(chunk.split())
        print(f"  chunk {chunk_num}: {word_count} words so far")
        chunk_num += 1
        

    return "\n\n".join(chunks)


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(prompts_path: str, model: str, output_path: str) -> None:
    if prompts_path.endswith(".parquet"):
        prompts_df = pd.read_parquet(prompts_path)
    else:
        prompts_df = pd.read_json(prompts_path, lines=True)

    records = []
    total   = len(prompts_df)

    for i, row in prompts_df.iterrows():
        story_id = row["id"]
        print(f"[{i+1}/{total}] Generating story id={story_id}")
        story = None
        error = None

        try:
            story = generate_story(row["prompt"], model, story_id)
        except Exception as e:
            error = str(e)
            print(f"  ERROR: {e}")

        records.append({
            "id":                   story_id,
            "language":             row.get("language", "en"),
            "task_type":            row.get("task_type", "generation"),
            "prompt":               row["prompt"],
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "generated_story":      story,
            "generation_error":     error,
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(records).to_parquet(output_path, index=False)
    print(f"\nDone. {len(records)} stories → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ConStory-Bench Mem0 generation pipeline"
    )
    parser.add_argument(
        "--prompts", required=True,
        help="Path to ConStory prompts file (.parquet or .jsonl)",
    )
    parser.add_argument(
        "--model", required=True,
        help="Anthropic model name, e.g. claude-haiku-4-5-20251001",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .parquet path, e.g. data/stories/my_model.parquet",
    )
    args = parser.parse_args()

    run_batch(args.prompts, args.model, args.output)