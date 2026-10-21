"""CR demo video (Manim, in the style of 3Blue1Brown). Needs Manim Community 0.22 (no LaTeX);
render from the repo root:

    python3 -m venv portfolio/.venv && portfolio/.venv/bin/pip install manim
    portfolio/.venv/bin/manim -qh portfolio/cr_video.py CRVideo --media_dir portfolio/media

The config below pins 1920x1080 at 60 fps. Numbers come from the stored benchmark curves, the v3
router's real outputs on two held-out benchmark prompts, and a live two-turn session.
"""
import json
from pathlib import Path

from manim import *

config.pixel_width, config.pixel_height, config.frame_rate = 1920, 1080, 60
config.background_color = BLACK

REPO = Path(__file__).resolve().parents[1]
SERIF, MONO = "Palatino", "Menlo"  # Palatino stands in for Computer Modern (no LaTeX here)
HAIKU, SONNET, OPUS = GREEN_C, BLUE_C, PURPLE_B
SAVE, RISK, KNOB = GOLD_C, RED_C, YELLOW_C
DIM = GREY_B


def T(s, size=32, color=WHITE, italic=False, **kw):
    return Text(s, font=SERIF, font_size=size, color=color, slant=ITALIC if italic else NORMAL, **kw)


def M(markup, size=32, color=WHITE):
    return MarkupText(markup, font=SERIF, font_size=size, color=color)


def span(s, color):
    return f'<span fgcolor="{ManimColor(color).to_hex()}">{s}</span>'


def part(m, plain, sub):
    """Glyphs of `sub` inside text `m` whose plain string is `plain` (glyphs skip spaces)."""
    flat, key = plain.replace(" ", ""), sub.replace(" ", "")
    i = flat.index(key)
    return VGroup(*m[i:i + len(key)])


def frontier(name):
    """Upper frontier of held-out (quality, total-bill savings), walking down from 100% quality."""
    c = json.loads((REPO / f"src/cheapshot/curves/{name}__cheapshot-v3.json").read_text())
    pts = sorted({(p["test"]["quality"], p["test"]["savings"]) for p in c["points"]}, reverse=True)
    out = [pts[0]]
    for q, s in pts[1:]:
        if s > out[-1][1]:
            out.append((q, s))
    return [(q, s) for q, s in out if q >= 0.8]


class CRVideo(Scene):
    def construct(self):
        self.title()
        self.threshold()
        self.cost_rule()
        self.benchmark()
        self.outro()

    def clear_all(self, t=0.45):
        self.play(LaggedStart(*[FadeOut(m) for m in self.mobjects], lag_ratio=0.03), run_time=t)
        self.clear()

    # ------------------------------------------------------------------ 1. title
    def title(self):
        cr = T("CR", 150).shift(UP * 1.6)
        sub = T("Cost-aware routing for Claude Code", 44).next_to(cr, DOWN, buff=0.45)
        rule = Line(LEFT, RIGHT, color=DIM, stroke_width=2).set_width(sub.width + 0.5).next_to(sub, DOWN, buff=0.28)
        tag = T("each turn goes to the cheapest Claude model that can handle it", 28, DIM).next_to(rule, DOWN, buff=0.3)
        names = VGroup(T("Haiku", 38, HAIKU), T("Sonnet", 38, SONNET), T("Opus", 38, OPUS)).arrange(RIGHT, buff=1.1, aligned_edge=UP)  # tops, not centers: "p" descends
        names.next_to(tag, DOWN, buff=0.6)
        self.play(Write(cr), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.2), GrowFromCenter(rule), run_time=0.7)
        self.play(FadeIn(tag, shift=UP * 0.1),
                  LaggedStart(*[FadeIn(n, shift=UP * 0.2) for n in names], lag_ratio=0.25), run_time=0.8)
        self.wait(0.6)
        self.clear_all()

    # ------------------------------------------------------------------ 2. risk threshold
    def threshold(self):
        eps = 0.38
        # cheapshot-v3 on two held-out prompts: P(model falls short) = P(a stronger model is needed)
        examples = [("“What is the difference between a guitar and a ukulele?”", (0.59, 0.28, 0.0), 1),
                    ("“How to code an api in python to download and chat with a huggingface llm?”", (0.94, 0.69, 0.0), 2)]
        y0, H, bw, xs = -1.75, 3.5, 1.15, (-5.3, -3.3, -1.3)
        cols, names = (HAIKU, SONNET, OPUS), ("Haiku", "Sonnet", "Opus")
        v, dim = [ValueTracker(0) for _ in xs], ValueTracker(0)

        def bar(i):
            h = max(v[i].get_value() * H, 0.02)
            over = min(max((v[i].get_value() - eps) / 0.05, 0), 1)
            return Rectangle(width=bw, height=h, stroke_color=cols[i], stroke_width=2, fill_color=cols[i],
                             fill_opacity=0.8 - 0.6 * dim.get_value() * over).move_to([xs[i], y0 + h / 2, 0])

        def val(i):
            t = T(f"{v[i].get_value():.2f}", 26, cols[i]).move_to([xs[i], y0 + v[i].get_value() * H + 0.28, 0])
            return t.add_background_rectangle(color=BLACK, opacity=1, buff=0.05).set_z_index(2)

        prompt = T(examples[0][0], 30, italic=True).move_to(UP * 3.2)
        cap = T("chance each model falls short, as estimated by the router", 24, DIM)
        cap.move_to([xs[0] - 1.0, 2.35, 0], aligned_edge=LEFT)
        base = Line([xs[0] - 1.0, y0, 0], [xs[2] + 1.0, y0, 0], color=DIM, stroke_width=2)
        labels = VGroup(*[VGroup(T(n, 30, c).move_to([x, y0 - 0.22, 0], aligned_edge=UP),  # tops align: "p" descends
                                 T(r, 21, DIM).move_to([x, y0 - 1.1, 0], aligned_edge=DOWN))  # every note ends in a descending "p"
                          for n, c, r, x in zip(names, cols, ("40× cheaper", "2× cheaper", "full price"), xs)])
        bars = VGroup(*[always_redraw(lambda i=i: bar(i)) for i in range(3)])
        vals = VGroup(*[always_redraw(lambda i=i: val(i)) for i in range(3)])

        self.play(FadeIn(prompt, shift=DOWN * 0.15), run_time=0.6)
        self.play(Create(base), FadeIn(labels, shift=UP * 0.1), FadeIn(cap), run_time=0.6)
        self.add(bars, vals)
        self.play(*[v[i].animate.set_value(x) for i, x in enumerate(examples[0][1])], run_time=0.8, rate_func=smooth)

        line = DashedLine([xs[0] - 1.0, y0 + eps * H, 0], [xs[2] + 1.0, y0 + eps * H, 0], color=KNOB, dash_length=0.12)
        eps_lab = VGroup(T("ε = 0.38", 32, KNOB, t2s={"ε": ITALIC}), T("risk you accept", 21, DIM)).arrange(DOWN, buff=0.1)
        eps_lab.next_to(line, RIGHT, buff=0.3)
        self.play(Create(line), FadeIn(eps_lab, shift=LEFT * 0.15), run_time=0.6)

        def decision(k):
            g = VGroup(T("→", 46), T(names[k], 52, cols[k])).arrange(RIGHT, buff=0.3)
            return g.move_to([4.7, 0.35, 0])

        dec = decision(examples[0][2])
        why = T("cheapest model under the line", 22, DIM).next_to(dec, DOWN, buff=0.3)
        box = SurroundingRectangle(labels[examples[0][2]][0], color=KNOB, buff=0.09)
        self.play(dim.animate.set_value(1), Create(box), FadeIn(dec, shift=RIGHT * 0.2), FadeIn(why), run_time=0.8)
        self.wait(0.6)

        p2 = T(examples[1][0], 27, italic=True).move_to(prompt)
        dec2 = decision(examples[1][2])
        self.play(FadeTransform(prompt, p2), *[v[i].animate.set_value(x) for i, x in enumerate(examples[1][1])],
                  Transform(box, SurroundingRectangle(labels[examples[1][2]][0], color=KNOB, buff=0.09)),
                  Transform(dec, dec2), run_time=1.1, rate_func=smooth)
        self.wait(0.3)
        bottom = T("Pick ε once; CR sends every turn to the cheapest model under it.", 28,
                   t2c={"ε": KNOB}, t2s={"ε": ITALIC}).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(bottom, shift=UP * 0.15), run_time=0.6)
        self.wait(0.9)
        self.clear_all()

    # ------------------------------------------------------------------ 3. cost-aware rule
    def cost_rule(self):
        head = T("Cost-aware rule", 42).to_edge(UP, buff=0.45)
        sub = T("score every model; the turn goes to the highest", 26, DIM).next_to(head, DOWN, buff=0.18)
        plain = "score(t) = savings(t) − λ · risk(t)"
        eq = M(f"score(<i>t</i>) = {span('savings(<i>t</i>)', SAVE)} − {span('<i>λ</i>', KNOB)} · "
               f"{span('risk(<i>t</i>)', RISK)}", 58).shift(UP * 0.45)
        sav, lam, risk = part(eq, plain, "savings(t)"), part(eq, plain, "λ"), part(eq, plain, "risk(t)")
        b1, b2 = Brace(sav, DOWN, buff=0.15, color=SAVE), Brace(risk, DOWN, buff=0.15, color=RISK)
        l1 = M("est. cost × (1 − price(<i>t</i>) ÷ price(Opus))", 24, SAVE).next_to(b1, DOWN, buff=0.15)
        l2 = M("P(<i>t</i> falls short)", 24, RISK).next_to(b2, DOWN, buff=0.15)
        knob = T("your quality knob", 26, KNOB).next_to(lam, UP, buff=0.75)
        arrow = Arrow(knob.get_bottom(), lam.get_top(), buff=0.1, color=KNOB, stroke_width=3,
                      max_tip_length_to_length_ratio=0.3)
        note = VGroup(
            T("est. cost comes from the router’s second answer: how long will the reply be?", 24, DIM,
              t2c={"how long will the reply be?": WHITE}),
            T("most of a bill sits in a few long answers, so that is where cheaper models pay off", 24, DIM,
              italic=True)).arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.45)

        self.play(FadeIn(head, shift=DOWN * 0.15), FadeIn(sub, shift=DOWN * 0.15), run_time=0.6)
        self.play(Write(eq), run_time=1.2)
        self.play(GrowFromCenter(b1), GrowFromCenter(b2), FadeIn(l1, shift=UP * 0.1), FadeIn(l2, shift=UP * 0.1),
                  run_time=0.8)
        self.play(GrowArrow(arrow), FadeIn(knob, shift=DOWN * 0.1), Indicate(lam, color=KNOB, scale_factor=1.3),
                  run_time=0.6)
        self.play(LaggedStart(*[FadeIn(n, shift=UP * 0.1) for n in note], lag_ratio=0.3), run_time=0.7)
        self.wait(1.2)
        self.clear_all()

    # ------------------------------------------------------------------ 4. benchmark
    def benchmark(self):
        head = T("205 held-out prompts, every answer judged against Opus", 32).to_edge(UP, buff=0.4)
        ax = Axes(x_range=[0.80, 1.00, 0.05], y_range=[0, 0.55, 0.1], x_length=7.0, y_length=4.5, tips=False,
                  axis_config={"color": GREY_A, "stroke_width": 2, "tick_size": 0.06})
        ax.move_to(LEFT * 2.75 + DOWN * 0.45)
        grid = VGroup(*[Line(ax.c2p(0.8, y), ax.c2p(1.0, y)) for y in (0.1, 0.2, 0.3, 0.4, 0.5)],
                      *[Line(ax.c2p(x, 0), ax.c2p(x, 0.55)) for x in (0.85, 0.9, 0.95, 1.0)])
        grid.set_stroke(BLUE_D, 1, opacity=0.35)
        ylabels = VGroup(*[T(f"{y:.0%}", 20, GREY_A).next_to(ax.c2p(0.8, y), LEFT, buff=0.18) for y in (0, .1, .2, .3, .4, .5)])
        xlabels = VGroup(*[T(f"{x:.0%}", 20, GREY_A).next_to(ax.c2p(x, 0), DOWN, buff=0.18) for x in (.8, .85, .9, .95, 1.0)])
        xt = T("quality: share of answers as good as Opus", 22, DIM).next_to(xlabels, DOWN, buff=0.2)
        yt = T("share of the bill saved", 22, DIM).next_to(ax, UP, buff=0.2).align_to(ylabels, LEFT)
        self.play(FadeIn(head, shift=DOWN * 0.15), run_time=0.5)
        self.play(Create(ax), Create(grid, lag_ratio=0.1), FadeIn(ylabels), FadeIn(xlabels), FadeIn(xt), FadeIn(yt),
                  run_time=0.8)

        curves = {}
        for name, col, lab, at in (("a_score", BLUE_C, "risk threshold", (0.872, 0.165)),
                                   ("cost_aware", KNOB, "cost-aware", (0.965, 0.40))):
            pts = frontier(name)
            line = VMobject(stroke_color=col, stroke_width=5).set_points_as_corners([ax.c2p(q, s) for q, s in pts])
            label = T(lab, 26, col).move_to(ax.c2p(*at))
            curves[name] = pts
            self.play(Create(line), FadeIn(label, shift=LEFT * 0.1), run_time=1.0, rate_func=smooth)

        # a dot rides the cost-aware frontier down to the 93.7%-quality point
        pts = curves["cost_aware"]
        k = min(range(len(pts)), key=lambda j: abs(pts[j][0] - 0.937))
        path = VMobject().set_points_as_corners([ax.c2p(q, s) for q, s in pts[:k + 1]])
        dot = Dot(path.get_start(), radius=0.09, color=KNOB)
        drops = always_redraw(lambda: VGroup(
            DashedLine(dot.get_center(), ax.c2p(ax.p2c(dot.get_center())[0], 0), color=KNOB, stroke_width=2, dash_length=0.08),
            DashedLine(dot.get_center(), ax.c2p(0.8, ax.p2c(dot.get_center())[1]), color=KNOB, stroke_width=2, dash_length=0.08)))
        self.add(drops, dot)
        self.play(MoveAlongPath(dot, path), run_time=1.0, rate_func=smooth)
        self.play(Flash(dot, color=KNOB, line_length=0.18, flash_radius=0.22), run_time=0.4)

        stats = [(28, "%", ("of the bill saved", "at 93.7% quality"), KNOB),
                 (5, " pts", ("quality vs. random routing", "at the same model mix"), BLUE_C),
                 (72, "%", ("saved in a live", "two-turn session"), HAIKU)]
        for i, (n, unit, label, col) in enumerate(stats):
            t = ValueTracker(0)
            pos = RIGHT * 2.0 + UP * (1.65 - i * 1.65)
            big = always_redraw(lambda t=t, unit=unit, col=col, pos=pos:
                                T(("+" if unit == " pts" else "") + f"{t.get_value():.0f}{unit}", 60, col)
                                .move_to(pos, aligned_edge=LEFT))
            lab = Paragraph(*label, font=SERIF, font_size=22, line_spacing=0.8, alignment="left")
            lab.move_to(pos + DOWN * 0.42, aligned_edge=UL)
            self.add(big)
            self.play(t.animate.set_value(n), FadeIn(lab, shift=UP * 0.1), run_time=0.6, rate_func=smooth)
        self.wait(1.3)
        self.clear_all()

    # ------------------------------------------------------------------ 5. outro
    def outro(self):
        cr = T("CR", 140).shift(UP * 1.0)
        cmd = Text("$ cr", font=MONO, font_size=34, color=GREY_A).next_to(cr, DOWN, buff=0.5)
        l1 = T("Claude Code with a router in front, on your Pro/Max plan", 30).next_to(cmd, DOWN, buff=0.35)
        l2 = T("fine-tuned Laya router on Oracle Ampere · Opus-judged benchmark · Python", 24, DIM).next_to(l1, DOWN, buff=0.25)
        self.play(Write(cr), run_time=0.9)
        self.play(AddTextLetterByLetter(cmd, time_per_char=0.06), rate_func=linear)
        self.play(FadeIn(l1, shift=UP * 0.15), FadeIn(l2, shift=UP * 0.15), run_time=0.5)
        self.wait(1.4)
        self.play(FadeOut(VGroup(cr, cmd, l1, l2)), run_time=0.5)
