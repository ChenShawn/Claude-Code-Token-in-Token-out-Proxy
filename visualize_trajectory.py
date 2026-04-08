#!/usr/bin/env python3
"""Console visualizer for trajectory JSONL files.

Usage:
    python visualize_trajectory.py data/*.jsonl
    python visualize_trajectory.py data/xxx.jsonl --turn 3
    python visualize_trajectory.py data/xxx.jsonl --turn 3 --full
    python visualize_trajectory.py data/xxx.jsonl --summary
"""

import argparse
import json
import os
import sys
import textwrap

# ── ANSI colors ──────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

FG_RED = "\033[31m"
FG_GREEN = "\033[32m"
FG_YELLOW = "\033[33m"
FG_BLUE = "\033[34m"
FG_MAGENTA = "\033[35m"
FG_CYAN = "\033[36m"
FG_WHITE = "\033[37m"
FG_GRAY = "\033[90m"

BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_RED = "\033[41m"

ROLE_STYLES = {
    "system": (FG_GRAY, "SYSTEM"),
    "user": (FG_GREEN + BOLD, "USER"),
    "assistant": (FG_BLUE + BOLD, "ASSISTANT"),
    "tool": (FG_YELLOW, "TOOL RESULT"),
}


def get_terminal_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 100


def hr(char="─", color=DIM):
    w = get_terminal_width()
    return f"{color}{char * w}{RESET}"


def truncate(text, max_len=200):
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def wrap_text(text, indent=4, width=None):
    if width is None:
        width = get_terminal_width() - indent - 2
    lines = text.split("\n")
    result = []
    prefix = " " * indent
    for line in lines:
        if len(line) <= width:
            result.append(prefix + line)
        else:
            wrapped = textwrap.wrap(line, width=width)
            for w in wrapped:
                result.append(prefix + w)
    return "\n".join(result)


def extract_text_content(content):
    """Extract plain text from message content (string or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(f"[{block.get('type', 'unknown')}]")
        return "\n".join(parts)
    return str(content) if content else ""


def format_tool_call(tc, compact=False):
    """Format a single tool call for display."""
    func = tc.get("function", {})
    name = func.get("name", "?")
    args_str = func.get("arguments", "{}")
    tc_id = tc.get("id", "")

    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        args = args_str

    if compact:
        # One-line summary
        if isinstance(args, dict):
            arg_summary = ", ".join(
                f"{k}={truncate(str(v), 60)}" for k, v in args.items()
            )
        else:
            arg_summary = truncate(str(args), 80)
        return f"{FG_MAGENTA}{name}{RESET}({DIM}{arg_summary}{RESET})"
    else:
        # Multi-line detailed view
        lines = []
        lines.append(f"  {FG_MAGENTA}{BOLD}{name}{RESET}  {DIM}id={tc_id}{RESET}")
        if isinstance(args, dict):
            for k, v in args.items():
                v_str = str(v)
                if len(v_str) > 300:
                    v_str = v_str[:297] + "..."
                lines.append(f"    {FG_CYAN}{k}{RESET}: {v_str}")
        else:
            lines.append(f"    {truncate(str(args), 300)}")
        return "\n".join(lines)


def format_tool_call_full(tc):
    """Format a tool call with full arguments (no truncation)."""
    func = tc.get("function", {})
    name = func.get("name", "?")
    args_str = func.get("arguments", "{}")
    tc_id = tc.get("id", "")

    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        args = args_str

    lines = []
    lines.append(f"  {FG_MAGENTA}{BOLD}{name}{RESET}  {DIM}id={tc_id}{RESET}")
    if isinstance(args, dict):
        for k, v in args.items():
            v_str = str(v)
            lines.append(f"    {FG_CYAN}{k}{RESET}:")
            lines.append(wrap_text(v_str, indent=6))
    else:
        lines.append(wrap_text(str(args), indent=4))
    return "\n".join(lines)


# ── Display modes ────────────────────────────────────────────────────────────


def print_summary(records, filepath):
    """Print a compact summary table of all turns in a JSONL file."""
    fname = os.path.basename(filepath)
    print()
    print(f"{BOLD}{BG_BLUE}{FG_WHITE} TRAJECTORY FILE {RESET} {BOLD}{fname}{RESET}")
    print(f"{DIM}Records: {len(records)}{RESET}")
    print()

    # Table header
    w = get_terminal_width()
    fmt = "{:<5} {:<38} {:>5} {:>6} {:>8} {:>10} {:>12}"
    header = fmt.format(
        "Turn", "Agent ID", "Msgs", "Tools", "Tok(in)", "Tok(out)", "Time"
    )
    print(f"{BOLD}{header}{RESET}")
    print(hr("─"))

    for i, rec in enumerate(records):
        agent_id = rec.get("agent_id", "?")[:36]
        msgs = rec.get("messages", [])
        meta = rec.get("metadata", {})
        n_msgs = len(msgs)

        # Count tool calls
        n_tool_calls = sum(
            len(m.get("tool_calls", [])) for m in msgs if m.get("tool_calls")
        )

        prompt_tok = meta.get("total_prompt_tokens", "?")
        resp_tok = meta.get("total_response_tokens", "?")
        create_time = meta.get("create_time", "?")
        if isinstance(create_time, str) and "T" in create_time:
            create_time = create_time.split("T")[1][:8]

        row = fmt.format(i, agent_id, n_msgs, n_tool_calls, prompt_tok, resp_tok, create_time)
        print(row)

    print(hr("─"))
    print()


def print_turn(record, turn_idx, full=False):
    """Print a single turn (record) with detailed message view."""
    meta = record.get("metadata", {})
    agent_id = record.get("agent_id", "?")
    msgs = record.get("messages", [])

    print()
    print(hr("═"))
    print(
        f"{BOLD}{BG_CYAN}{FG_WHITE} TURN {turn_idx} {RESET}"
        f"  agent={FG_CYAN}{agent_id}{RESET}"
        f"  msgs={len(msgs)}"
        f"  prompt_tok={meta.get('total_prompt_tokens', '?')}"
        f"  resp_tok={meta.get('total_response_tokens', '?')}"
        f"  time={meta.get('create_time', '?')}"
    )
    print(hr("═"))

    # Group messages into logical steps (assistant action + tool results)
    step = 0
    for i, msg in enumerate(msgs):
        role = msg.get("role", "?")
        content = extract_text_content(msg.get("content", ""))
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "")
        style, label = ROLE_STYLES.get(role, (FG_WHITE, role.upper()))

        if role == "system":
            # System messages: just show summary
            content_len = len(content)
            print(f"\n{style}[{label}]{RESET} {DIM}({content_len} chars){RESET}")
            if full:
                print(wrap_text(content, indent=4))
            else:
                # Show first 2 lines
                lines = content.strip().split("\n")
                for line in lines[:2]:
                    print(f"    {DIM}{truncate(line, 120)}{RESET}")
                if len(lines) > 2:
                    print(f"    {DIM}... ({len(lines)} lines total){RESET}")

        elif role == "user":
            print(f"\n{hr('─', FG_GREEN + DIM)}")
            print(f"{style}[{label}]{RESET}")
            if content.strip():
                if full:
                    print(wrap_text(content, indent=4))
                else:
                    # Show content, truncate if very long
                    # Skip system-reminder tags for readability
                    display = content
                    # Try to extract meaningful user text from system-reminder blocks
                    import re
                    parts = re.split(r"<system-reminder>.*?</system-reminder>", display, flags=re.DOTALL)
                    clean = "\n".join(p.strip() for p in parts if p.strip())
                    if clean:
                        display = clean
                    if len(display) > 500 and not full:
                        print(wrap_text(truncate(display, 500), indent=4))
                    else:
                        print(wrap_text(display, indent=4))
            else:
                print(f"    {DIM}(empty){RESET}")

        elif role == "assistant":
            step += 1
            print(f"\n{hr('─', FG_BLUE + DIM)}")
            print(f"{style}[{label}  Step {step}]{RESET}")

            if content.strip():
                if full:
                    print(wrap_text(content, indent=4))
                else:
                    if len(content) > 500:
                        print(wrap_text(truncate(content, 500), indent=4))
                    else:
                        print(wrap_text(content, indent=4))

            if tool_calls:
                print(f"    {FG_MAGENTA}Tool Calls ({len(tool_calls)}):{RESET}")
                for tc in tool_calls:
                    if full:
                        print(format_tool_call_full(tc))
                    else:
                        print(format_tool_call(tc, compact=False))

            if not content.strip() and not tool_calls:
                print(f"    {DIM}(empty response){RESET}")

        elif role == "tool":
            name_hint = ""
            if tool_call_id:
                name_hint = f" {DIM}id={tool_call_id[:24]}...{RESET}"
            print(f"  {style}[{label}]{RESET}{name_hint}")
            if content.strip():
                if full:
                    print(wrap_text(content, indent=6))
                else:
                    # Truncate long tool results
                    if len(content) > 300:
                        print(wrap_text(truncate(content, 300), indent=6))
                    else:
                        print(wrap_text(content, indent=6))
            else:
                print(f"      {DIM}(empty){RESET}")

    print()
    print(hr("═"))


def print_conversation_flow(records):
    """Print a high-level conversation flow: just user messages and assistant text responses."""
    print()
    print(f"{BOLD}{BG_MAGENTA}{FG_WHITE} CONVERSATION FLOW {RESET}")
    print(hr("─"))

    for turn_idx, rec in enumerate(records):
        msgs = rec.get("messages", [])
        for msg in msgs:
            role = msg.get("role", "")
            content = extract_text_content(msg.get("content", ""))
            tool_calls = msg.get("tool_calls", [])

            if role == "user" and content.strip():
                import re
                parts = re.split(r"<system-reminder>.*?</system-reminder>", content, flags=re.DOTALL)
                clean = "\n".join(p.strip() for p in parts if p.strip())
                if clean:
                    print(f"\n  {FG_GREEN}{BOLD}USER:{RESET}")
                    print(wrap_text(truncate(clean, 300), indent=4))

            elif role == "assistant":
                if content.strip():
                    print(f"\n  {FG_BLUE}{BOLD}ASSISTANT:{RESET}")
                    print(wrap_text(truncate(content, 300), indent=4))
                if tool_calls:
                    calls_str = ", ".join(
                        format_tool_call(tc, compact=True) for tc in tool_calls
                    )
                    print(f"    {FG_MAGENTA}calls: {calls_str}{RESET}")

    print()
    print(hr("─"))


# ── Main ─────────────────────────────────────────────────────────────────────


def load_jsonl(filepath):
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def interactive_browse(records, filepath):
    """Interactive mode: browse turns one by one."""
    fname = os.path.basename(filepath)
    n = len(records)

    print_summary(records, filepath)

    while True:
        print(f"{BOLD}Commands:{RESET}")
        print(f"  {FG_CYAN}0-{n-1}{RESET}    View turn detail")
        print(f"  {FG_CYAN}f N{RESET}     View turn N with full content (no truncation)")
        print(f"  {FG_CYAN}s{RESET}       Show summary table")
        print(f"  {FG_CYAN}c{RESET}       Show conversation flow (user/assistant text only)")
        print(f"  {FG_CYAN}q{RESET}       Quit")
        print()

        try:
            cmd = input(f"{FG_YELLOW}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd.lower() == "q":
            break
        elif cmd.lower() == "s":
            print_summary(records, filepath)
        elif cmd.lower() == "c":
            print_conversation_flow(records)
        elif cmd.lower().startswith("f "):
            try:
                idx = int(cmd.split()[1])
                if 0 <= idx < n:
                    print_turn(records[idx], idx, full=True)
                else:
                    print(f"{FG_RED}Invalid turn index. Must be 0-{n-1}.{RESET}")
            except (ValueError, IndexError):
                print(f"{FG_RED}Usage: f <turn_number>{RESET}")
        else:
            try:
                idx = int(cmd)
                if 0 <= idx < n:
                    print_turn(records[idx], idx, full=False)
                else:
                    print(f"{FG_RED}Invalid turn index. Must be 0-{n-1}.{RESET}")
            except ValueError:
                print(f"{FG_RED}Unknown command: {cmd}{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Console visualizer for trajectory JSONL files"
    )
    parser.add_argument("files", nargs="+", help="JSONL file(s) to visualize")
    parser.add_argument(
        "--turn", "-t", type=int, default=None, help="Show a specific turn (0-indexed)"
    )
    parser.add_argument(
        "--full", "-f", action="store_true", help="Show full content without truncation"
    )
    parser.add_argument(
        "--summary", "-s", action="store_true", help="Show summary table only"
    )
    parser.add_argument(
        "--flow", "-c", action="store_true", help="Show conversation flow only"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive browsing mode"
    )

    args = parser.parse_args()

    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"{FG_RED}File not found: {filepath}{RESET}", file=sys.stderr)
            continue

        records = load_jsonl(filepath)

        if args.interactive:
            interactive_browse(records, filepath)
        elif args.summary:
            print_summary(records, filepath)
        elif args.flow:
            print_conversation_flow(records)
        elif args.turn is not None:
            if 0 <= args.turn < len(records):
                print_turn(records[args.turn], args.turn, full=args.full)
            else:
                print(
                    f"{FG_RED}Turn {args.turn} out of range (0-{len(records)-1}){RESET}",
                    file=sys.stderr,
                )
        else:
            # Default: interactive if tty, otherwise summary + all turns
            if sys.stdin.isatty():
                interactive_browse(records, filepath)
            else:
                print_summary(records, filepath)
                for i, rec in enumerate(records):
                    print_turn(rec, i, full=args.full)


if __name__ == "__main__":
    main()
