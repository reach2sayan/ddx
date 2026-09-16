#!/usr/bin/env python3
"""Inline snippets/ into a single self-contained index.html.

The snippets are the source of truth: they compile, and godbolt.json maps each
one to its Compiler Explorer link.  A code slide always carries its whole file --
nothing is cropped, nothing is elided.  A slide that is making one point about
the file marks the lines it is talking about, and the rest of the program stays
on screen at full strength beside them.
"""

import html
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
SNIPPETS = HERE / "snippets"

# ---------------------------------------------------------------- highlighting

KEYWORDS = {
    "auto", "char", "class", "const", "constexpr", "decltype", "double",
    "false", "int", "operator", "return", "sizeof", "static_assert", "struct",
    "template", "true", "using", "void",
}
TOKEN = re.compile(r"[A-Za-z_]\w*|\b\d+(?:\.\d+)?\b")



def paint_cpp(line):
    code, comment = line, ""
    cut = line.find("//")
    if cut >= 0:
        code, comment = line[:cut], line[cut:]

    def token(m):
        t = m.group(0)
        if t in KEYWORDS:
            return f'<span class="k">{t}</span>'
        if t[0].isdigit():
            return f'<span class="n">{t}</span>'
        return t

    out = TOKEN.sub(token, html.escape(code))
    if comment:
        out += f'<span class="c">{html.escape(comment)}</span>'
    return out


def paint_sh(line):
    if line.startswith("$"):
        return f'<span class="k">{html.escape(line)}</span>'
    cut = line.find("#")
    if cut >= 0:
        return (html.escape(line[:cut])
                + f'<span class="c">{html.escape(line[cut:])}</span>')
    return html.escape(line)


# ---------------------------------------------------------------- slide bodies

def code(file, part=None, lang="cpp", mark=None):
    """One contiguous part of a file.  The parts of a file have to tile it, in
    order, with nothing skipped -- check_deck() will not build a deck where a
    slide quietly drops a line.  A slide is only ever what it is talking about,
    so nothing on it needs marking."""
    return {"file": file, "part": part, "lang": lang, "mark": mark}


DECK = [
    {
        "kind": "title",
        "h1": "sizeof(x*y + 2*x) == 1",
        "lede": "Expression templates that never grow",
        "note": "lightning talk &middot; CppCon 2026",
        "build": "gcc 15.2 &middot; clang 18.1 &nbsp;&mdash;&nbsp; -std=c++23 -O2, "
                 "the same code on both",
        "alert": True,
        "colophon": "Slides produced with Claude (Anthropic). The idea, the code and the "
                    "measurements are the author's own.",
    },
    {
        "h2": "What an expression template is",
        "body": "<p>An expression template does not do the arithmetic when you write it. "
                "It <em>represents</em> the computation &mdash; and it represents it as a "
                "<em>type</em>:</p>",
        "code": code("00-the-tree.txt", lang="txt"),
        "step": '<p class="takeaway big">The structure is all in the type. So how many bytes '
                'does <code>e</code> need?</p>',
    },
    {
        "h2": "Attempt 1 &mdash; children by value",
        "code": code("01-by-value.cpp", (1, 27)),
        "step": '<p class="takeaway">Safe &mdash; every node owns its children. Which also '
                'means it owns a <em>copy</em>.</p>',
    },
    {
        "h2": "&hellip; and it grows",
        "code": code("01-by-value.cpp", (28, 41)),
        "step": '<p class="takeaway bad">Size is O(nodes) &mdash; a real model is kilobytes '
                'of expression &mdash; and every <code>operator+</code> copies the subtree '
                'it just built.</p>',
    },
    {
        "h2": "Attempt 2 &mdash; children by reference",
        "code": code("02-by-reference.cpp", (1, 27)),
        "step": '<p class="takeaway">A reference is 8 bytes whatever it points at.</p>',
    },
    {
        "h2": "&hellip; it watches, and then it dangles",
        "code": code("02-by-reference.cpp", (28, 40)),
        # measured, not remembered -- same file, same box
        "after": '<p class="matrix">exit code &nbsp; <b>gcc 15</b> &nbsp;-O0 &rarr; 20 '
                 '&nbsp;&middot;&nbsp; -O1 &rarr; 8 &nbsp;&middot;&nbsp; -O2 &rarr; 0 '
                 '&nbsp;&middot;&nbsp; -O3 warns, at last &nbsp;&nbsp;&nbsp; '
                 '<b>clang 18</b> &nbsp;20 at every level, silent.</p>',
        "step": '<p class="takeaway bad">A named subexpression outlives the temporaries it '
                'points at &mdash; right in debug, silently wrong in release.</p>',
    },
    {
        "h2": "Attempt 3 &mdash; the reference goes in the type (C++17)",
        "body": "<p>Attempt 2 put the reference in the <em>object</em>. Put it in the "
                "<em>type</em> instead &mdash; the node has no members at all:</p>",
        "code": code("03-in-the-type.cpp", (1, 29), mark=(13, 17)),
        "step": '<p class="takeaway">Nothing to copy, nothing to dangle.</p>',
    },
    {
        "h2": "&hellip; one byte, and it still evaluates",
        "code": code("03-in-the-type.cpp", (30, 47)),
        "step": '<p class="takeaway">One byte, no lifetime, and it still reads the live '
                'variable. But the 2 arrives as <code>Ref&lt;c2&gt;</code>: the type records '
                '<em>where</em> the value lives, not <em>what</em> it is, so the compiler '
                'cannot fold it.</p>',
    },
    {
        "h2": "&hellip; and the constant's value too (C++20)",
        "code": code("05-constant-in-the-type.cpp", (1, 28), mark=(14, 16)),
        "step": '<p class="takeaway good">C++20 takes a <code>double</code> as a template '
                'parameter, so the 2 needs no object to point at.</p>',
    },
    {
        "h2": "&hellip; and the 2 costs nothing either",
        "code": code("05-constant-in-the-type.cpp", (29, 43)),
        "step": '<p class="takeaway good">One byte for the whole expression, still '
                'reading the live variable &mdash; and the part that needs no variable is '
                '<code>constexpr</code>, so it folds before the program runs.</p>',
    },
    {
        "h2": "The trade",
        "two": (
            ("Gained", ["no lifetimes", "no copies", "no size",
                        "a fully inspectable tree"], "good"),
            ("Paid", ["an instantiation per expression", "leaves must be objects with static storage duration"], "bad"),
        ),
        "step": '<p class="takeaway big good mono">sizeof(x*y + 2*x) == 1</p>',
    },
    {
        "kind": "title",
        "h1": "Thank you",
        "lede": "",
        "note": "every slide carries its whole program &mdash; the link under each one "
                "opens it on Compiler Explorer",
    },
]


# ---------------------------------------------------------------- page assembly

def render_code(spec):
    path = SNIPPETS / spec["file"]
    lines = path.read_text().rstrip("\n").split("\n")
    first, last = spec["part"] or (1, len(lines))
    paint = paint_sh if spec["lang"] == "sh" else paint_cpp
    numbered = spec["lang"] != "txt"

    mark = set(range(spec["mark"][0], spec["mark"][1] + 1)) if spec["mark"] else set()
    rows = []
    for n in range(first, last + 1):
        gutter = f'<span class="ln">{n}</span>' if numbered else ""
        cls = "line here" if n in mark else "line"
        rows.append(f'<span class="{cls}">{gutter}{paint(lines[n - 1]) or "&nbsp;"}</span>')
    while rows and not lines[first + len(rows) - 2].strip():
        rows.pop()                         # never end a listing on a blank line
    return f'<pre class="code {spec["lang"]}"><code>{"".join(rows)}</code></pre>'


def render_link(spec, links):
    """The listing's handle, set beside the slide title.  No filename: nobody in
    the room has the files, so a listing with no link says nothing at all."""
    href = links.get(spec["file"])
    if not href:
        return ""
    lines = (SNIPPETS / spec["file"]).read_text().rstrip("\n").split("\n")
    first, last = spec["part"] or (1, len(lines))
    whole = (f"whole program, {len(lines)} lines" if (first, last) == (1, len(lines))
             else f"lines {first}&ndash;{last} of {len(lines)}")
    return (f'<span class="handle">'
            f'<a class="ce" href="{href}" rel="noopener" target="_blank">{href}</a>'
            f'<span class="whole">{whole}</span></span>')


ALERT = (
    '<div class="alert">'
    '<span class="siren">&#9888;</span>'
    '<span><b>Slideware &mdash; do not try this (exactly) at home.</b> '
    'Every listing compiles and every number on it is real, but it is cut to fit '
    'a slide.</span>'
    "</div>"
)


def render_two(two):
    cols = []
    for head, items, tone in two:
        mark = "&#10003;" if tone == "good" else "&#10007;"
        rows = "".join(f'<p class="{tone}"><span class="mark">{mark}</span>{i}</p>'
                       for i in items)
        cols.append(f"<div><h3>{head}</h3>{rows}</div>")
    return f'<div class="two">{"".join(cols)}</div>'


def render(deck, links):
    out = []
    for i, slide in enumerate(deck):
        if slide.get("kind") == "title":
            parts = [f'<h1 class="mono">{slide["h1"]}</h1>']
            if slide["lede"]:
                parts.append(f'<p class="lede">{slide["lede"]}</p>')
            parts.append(f'<p class="note">{slide["note"]}</p>')
            if slide.get("build"):
                parts.append(f'<p class="build">{slide["build"]}</p>')
            if slide.get("alert"):
                parts.append(ALERT)
            if slide.get("colophon"):
                parts.append(f'<p class="colophon">{slide["colophon"]}</p>')
            out.append(f'<section class="slide title"><div class="pad">'
                       f'{"".join(parts)}</div></section>')
            continue
        head = render_link(slide["code"], links) if slide.get("code") else ""
        parts = [f'<h2>{slide["h2"]}{head}</h2>']
        if slide.get("alert"):
            parts.append(ALERT)
        if slide.get("body"):
            parts.append(slide["body"])
        if slide.get("code"):
            parts.append(render_code(slide["code"]))
        if slide.get("after"):
            parts.append(slide["after"])
        if slide.get("two"):
            parts.append(render_two(slide["two"]))
        if slide.get("step"):
            parts.append(slide["step"])
        out.append(f'<section class="slide"><div class="pad">'
                   f'{"".join(parts)}</div></section>')
    return "\n".join(out)


def check_deck(deck):
    """The parts of a file must tile it in order: no gap, no overlap, nothing
    past the end.  This is the whole guarantee that a slide never drops code."""
    bad = []
    parts = {}
    for slide in deck:
        spec = slide.get("code")
        if spec:
            parts.setdefault(spec["file"], []).append(spec)
    for path in sorted(SNIPPETS.glob("*")):
        if path.name == "godbolt.json":
            continue
        if path.name not in parts:
            bad.append(f"{path.name}: on no slide")
            continue
        lines = path.read_text().rstrip("\n").split("\n")
        total = len(lines)
        want = 1
        for spec in parts[path.name]:
            first, last = spec["part"] or (1, total)
            if first != want:
                bad.append(f"{path.name}: part starts at {first}, expected {want}")
            if not lines[first - 1].strip():
                bad.append(f"{path.name}: part {first}-{last} opens on a blank line")
            if last < first:
                bad.append(f"{path.name}: part {first}-{last} runs backwards")
            want = last + 1
        if want != total + 1:
            bad.append(f"{path.name}: {total - want + 1} lines past the last part")
    return bad


CSS = """
/* Okabe-Ito, on a dark ground: blue/orange/grey only, never red vs green.
   Every coloured cue is doubled by a glyph, a weight or an italic. */
:root {
  --bg: #0f141b;
  --panel: #19212b;
  --rule: #33404f;
  --ink: #edf1f7;
  --muted: #aab7c8;
  --blue: #63bdf0;
  --orange: #eaa62b;
  --grey: #9dabbd;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; height: 100%; background: var(--bg); color: var(--ink);
  font-family: "Inter", "Segoe UI", system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.deck { height: 100%; }
.slide {
  display: none; overflow: hidden; flex-direction: column; justify-content: center;
  height: 100%; padding: 3.5vh 5vw 5vh; font-size: clamp(15px, 1.7vw, 26px);
}
.pad { width: 100%; }
.slide.on { display: flex; }
h1 { font-size: 2.6em; font-weight: 700; margin: 0 0 .4em; letter-spacing: -.01em; }
h2 {
  display: flex; align-items: baseline; justify-content: space-between; gap: 1.5em;
  font-size: 1.3em; font-weight: 650; margin: 0 0 .5em; color: var(--ink);
  border-bottom: 2px solid var(--rule); padding-bottom: .35em;
}
.handle {
  flex: none; text-align: right; font-size: .44em; font-weight: 400; line-height: 1.45;
  font-family: "JetBrains Mono", ui-monospace, monospace; color: var(--muted);
}
.handle .ce { display: block; color: var(--blue); text-decoration: underline; }
.handle .ce:hover { color: var(--orange); }
.handle .whole { display: block; }
h3 {
  font-size: .72em; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; color: var(--muted); margin: 0 0 .7em;
}
p, li { line-height: 1.55; margin: 0 0 .5em; }
ul { margin: 0 0 .6em 1.1em; padding: 0; }
li { margin-bottom: .35em; }
em { font-style: normal; font-weight: 700; color: var(--blue); }
.title { align-items: center; text-align: center; }
.lede { font-size: 1.15em; color: var(--muted); }
.note { font-size: .8em; color: var(--muted); }
.mono, code, pre { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
p code, li code {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 4px;
  padding: .05em .35em; font-size: .88em;
}

/* ---------- code ---------- */
pre.code {
  margin: 0; padding: .7em 0; background: var(--panel);
  border: 1px solid var(--rule); border-radius: 8px;
  overflow: hidden; line-height: 1.4; font-size: .95em;
  white-space: pre; tab-size: 2;
}
pre.code .line { display: block; padding: 0 1em 0 .9em; }
pre.code .ln {
  display: inline-block; width: 2.4em; text-align: right; margin-right: 1em;
  color: #67748a; user-select: none;
}
pre.code .k { color: var(--blue); font-weight: 600; }
/* The one new idea in the deck, marked in place: a rule, a tint and extra
   weight, so it reads as marked with no colour at all. */
pre.code .here {
  background: #22303e; border-left: 3px solid var(--blue);
  margin-left: -.9em; padding-left: calc(.9em - 3px); font-weight: 600;
}
pre.code .here .c { color: #b9c6d8; }
pre.code .n { color: var(--orange); }
pre.code .c { color: var(--grey); font-style: italic; }


/* ---------- slideware alert ---------- */
/* Orange, a siren, a rule and an upright weight -- four cues, so it still reads
   as a warning with no colour at all. */
.alert {
  display: flex; gap: .7em; align-items: flex-start; text-align: left;
  margin: 1.2em auto 0; max-width: 46em; padding: .7em .9em;
  font-size: .68em; line-height: 1.5; color: var(--ink);
  background: #2a2317; border: 1px solid var(--orange);
  border-left: 4px solid var(--orange); border-radius: 8px;
}
.alert .siren { font-size: 1.2em; line-height: 1.2; color: var(--orange); }
.alert b { color: var(--orange); }
.slide:not(.title) .alert { margin: 0 0 .8em; max-width: none; }

.colophon {
  margin: 1em 0 0; font-size: .58em; color: var(--muted); font-style: italic;
}

.build {
  margin: .9em 0 0; font-size: .66em; color: var(--muted);
  font-family: "JetBrains Mono", ui-monospace, monospace;
}

.matrix {
  margin: .5em 0 0; font-size: .64em; color: var(--muted);
  font-family: "JetBrains Mono", ui-monospace, monospace;
}
.matrix b { color: var(--ink); font-weight: 700; }

/* ---------- takeaways ---------- */
.takeaway {
  margin: .7em 0 0; padding-left: .8em; border-left: 3px solid var(--rule);
  color: var(--ink);
}
.takeaway.big { font-size: 1.25em; font-weight: 650; }
.takeaway.note { font-size: .78em; color: var(--muted); }
.takeaway.good { border-left-color: var(--blue); }
.takeaway.good::before { content: "\\2713\\00a0"; color: var(--blue); font-weight: 700; }
.takeaway.bad { border-left-color: var(--orange); }
.takeaway.bad::before { content: "\\2717\\00a0"; color: var(--orange); font-weight: 700; }

/* ---------- two columns ---------- */
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin-top: .8em; }
.two > div {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 8px;
  padding: .9em 1.1em .7em;
}
.two p { margin-bottom: .35em; font-size: .92em; }
.two .mark { display: inline-block; width: 1.4em; font-weight: 700; }
.two .good .mark { color: var(--blue); }
.two .bad .mark { color: var(--orange); }
.two .bad { font-style: italic; }

/* ---------- chrome ---------- */
.bar { position: fixed; left: 0; bottom: 0; height: 3px; background: var(--blue); }
.count {
  position: fixed; right: 1.2em; bottom: 1em; font-size: 12px; color: var(--muted);
  font-family: "JetBrains Mono", ui-monospace, monospace;
}
@media print {
  .slide { display: flex !important; page-break-after: always; height: 100vh; }
  .bar, .count { display: none; }
}
"""

JS = """
const slides = [...document.querySelectorAll(".slide")];
const bar = document.querySelector(".bar");
const count = document.querySelector(".count");
let at = 0;

const draw = () => {
  slides.forEach((s, i) => s.classList.toggle("on", i === at));
  bar.style.width = ((at + 1) / slides.length) * 100 + "%";
  count.textContent = at + 1 + " / " + slides.length;
  if (location.hash !== "#" + (at + 1)) history.replaceState(null, "", "#" + (at + 1));
  fit();
};

// Slideware, not a reader: no listing may scroll, in either direction.  Shrink
// it until the whole program fits the slide, top to bottom and edge to edge.
const fit = () => {
  const slide = slides[at];
  const pad = slide.querySelector(".pad");
  const pre = slide.querySelector("pre.code");
  const style = getComputedStyle(slide);
  const room = slide.clientHeight - parseFloat(style.paddingTop)
             - parseFloat(style.paddingBottom);
  pad.style.fontSize = "";
  if (pre) {
    const over = () => pad.scrollHeight > room + 1 || pre.scrollWidth > pre.clientWidth + 1;
    let size = 0.95;
    pre.style.fontSize = size + "em";
    while (size > 0.3 && over()) {
      size -= 0.01;
      pre.style.fontSize = size + "em";
    }
    pre.dataset.fit = size.toFixed(2);
    return;
  }
  // No listing to shrink: scale the slide itself, so a dense table never clips.
  let size = 1;
  while (size > 0.55 && pad.scrollHeight > room + 1) {
    size -= 0.02;
    pad.style.fontSize = size + "em";
  }
};

// One key, one slide.  Nothing is revealed in pieces.
const go = (d) => {
  at = Math.min(Math.max(at + d, 0), slides.length - 1);
  draw();
};

addEventListener("keydown", (e) => {
  const fwd = ["ArrowRight", "ArrowDown", "PageDown", " ", "Enter"];
  const back = ["ArrowLeft", "ArrowUp", "PageUp", "Backspace"];
  if (fwd.includes(e.key)) { e.preventDefault(); go(1); }
  else if (back.includes(e.key)) { e.preventDefault(); go(-1); }
  else if (e.key === "Home") { at = 0; draw(); }
  else if (e.key === "End") { at = slides.length - 1; draw(); }
  else if (e.key === "f") document.documentElement.requestFullscreen?.();
});
addEventListener("resize", fit);
addEventListener("click", (e) => { if (!e.target.closest("a")) go(1); });

at = Math.min(Math.max(parseInt(location.hash.slice(1), 10) || 1, 1), slides.length) - 1;
draw();
"""

PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>sizeof(x*y + 2*x) == 1</title>
    <style>{css}</style>
  </head>
  <body>
    <div class="deck">
{slides}
    </div>
    <div class="bar"></div>
    <div class="count"></div>
    <script>{js}</script>
  </body>
</html>
"""


def main():
    problems = check_deck(DECK)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    links = json.loads((SNIPPETS / "godbolt.json").read_text())
    page = PAGE.format(css=CSS, js=JS, slides=render(DECK, links))
    (HERE / "index.html").write_text(page)
    print(f"index.html: {len(DECK)} slides, {len(page)} bytes, no external files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
