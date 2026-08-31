"""Manim explainer: equitable partitions on an 8-neuron, 3-role network.

The exact case animated here:

    role 1 = {1, 2}      role 2 = {3, 4, 5}      role 3 = {6, 7, 8}

    edges: 1-2, complete bipartite {1,2}x{3,4,5},
           matching 3-6, 4-7, 5-8, triangle 6-7-8

The partition is equitable: every neuron in role i has exactly B[i, j]
neighbours in role j, with divisor matrix

    B = [[1, 3, 0],
         [2, 0, 1],
         [0, 1, 2]]

so A H = H B for the 8x3 indicator (lift) matrix H, the role subspace
col(H) is A-invariant, and the 8-dimensional dynamics restricted to it
are exactly the 3-dimensional quotient dynamics y -> B y.

Render:
    manim -qm --save_sections src/flexsym/anim/equitable_partition.py EquitablePartitionStory
    manim -ql  src/flexsym/anim/equitable_partition.py EquitablePartitionStory   # fast draft
"""

from __future__ import annotations

import numpy as np
from manim import *

# --------------------------------------------------------------------------
# the exact network
# --------------------------------------------------------------------------

CLUSTERS = {0: [1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
CLUSTER_OF = {n: k for k, ns in CLUSTERS.items() for n in ns}

EDGES = [
    (1, 2),
    (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5),
    (3, 6), (4, 7), (5, 8),
    (6, 7), (7, 8), (6, 8),
]

A = np.zeros((8, 8))
for _u, _v in EDGES:
    A[_u - 1, _v - 1] = A[_v - 1, _u - 1] = 1.0

H = np.zeros((8, 3))
for _k, _ns in CLUSTERS.items():
    for _n in _ns:
        H[_n - 1, _k] = 1.0

B = np.array([[1.0, 3.0, 0.0],
              [2.0, 0.0, 1.0],
              [0.0, 1.0, 2.0]])

assert np.allclose(A @ H, H @ B), "partition is not equitable"

NBRS = {n: [m for m in range(1, 9) if A[n - 1, m - 1]] for n in range(1, 9)}


def counts(n: int) -> tuple[int, int, int]:
    """Neighbours of neuron ``n`` in each of the three roles."""
    c = [0, 0, 0]
    for m in NBRS[n]:
        c[CLUSTER_OF[m]] += 1
    return tuple(c)


POS = {
    1: np.array([-4.6, 1.1, 0.0]), 2: np.array([-4.6, -1.1, 0.0]),
    3: np.array([-1.6, 2.2, 0.0]), 4: np.array([-1.6, 0.0, 0.0]), 5: np.array([-1.6, -2.2, 0.0]),
    6: np.array([1.6, 2.2, 0.0]), 7: np.array([1.6, 0.0, 0.0]), 8: np.array([1.6, -2.2, 0.0]),
}

C1, C2, C3 = "#4C8DF5", "#F2A03D", "#4FBF7F"
CCOL = {0: C1, 1: C2, 2: C3}
NODE_COL = {n: CCOL[CLUSTER_OF[n]] for n in range(1, 9)}
BAR_COL = [NODE_COL[n] for n in range(1, 9)]
EDGE_BASE = GREY_B

# --------------------------------------------------------------------------
# mobjects
# --------------------------------------------------------------------------


class NeuronGraph(VGroup):
    """The 8-neuron network, with per-node and per-edge handles."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dots, self.labels, self.edges = {}, {}, {}

        for u, v in EDGES:
            if (u, v) == (6, 8):
                e = ArcBetweenPoints(POS[6], POS[8], angle=-1.0)
            else:
                e = Line(POS[u], POS[v])
            e.set_stroke(EDGE_BASE, 3, opacity=0.7)
            self.edges[(u, v)] = e

        for n in range(1, 9):
            d = Circle(radius=0.30).move_to(POS[n])
            d.set_fill(GREY_D, 1.0).set_stroke(WHITE, 2.5)
            self.dots[n] = d
            self.labels[n] = Text(str(n), font_size=24, weight=BOLD,
                                  color=BLACK).move_to(POS[n])

        self.edge_group = VGroup(*self.edges.values()).set_z_index(0)
        self.node_group = VGroup(*[VGroup(self.dots[n], self.labels[n])
                                   for n in range(1, 9)]).set_z_index(2)
        self.add(self.edge_group, self.node_group)

    def incident(self, n: int) -> list[Mobject]:
        return [e for (u, v), e in self.edges.items() if n in (u, v)]


class BarPanel(VGroup):
    """Signed bar chart with a fixed number of slots."""

    def __init__(self, colors, width=0.36, gap=0.14, unit=1.15, group_sizes=None, **kwargs):
        super().__init__(**kwargs)
        self.colors, self.w, self.unit = list(colors), width, unit
        n = len(self.colors)
        pitch = width + gap
        extra = 0.0
        self.xs = []
        if group_sizes is None:
            group_sizes = [n]
        idx = 0
        for gi, gs in enumerate(group_sizes):
            for _ in range(gs):
                self.xs.append(idx * pitch + extra)
                idx += 1
            extra += 0.30
        span = max(self.xs) - min(self.xs)
        self.xs = [x - min(self.xs) - span / 2 for x in self.xs]

        self.baseline = Line([self.xs[0] - pitch / 2, 0, 0], [self.xs[-1] + pitch / 2, 0, 0])
        self.baseline.set_stroke(GREY_B, 2)
        self.bars = self._target(np.zeros(n))
        self.add(self.baseline, self.bars)

    def _target(self, vals) -> VGroup:
        origin = self.baseline.get_center()
        grp = VGroup()
        for i, v in enumerate(vals):
            h = max(abs(float(v)) * self.unit, 0.02)
            r = Rectangle(width=self.w, height=h)
            r.set_fill(self.colors[i], 0.92).set_stroke(width=0)
            sgn = 1.0 if v >= 0 else -1.0
            r.move_to(origin + np.array([self.xs[i], sgn * h / 2, 0.0]))
            grp.add(r)
        return grp

    def set_values(self, vals):
        self.bars.become(self._target(vals))
        return self

    def anim(self, vals, **kw):
        return Transform(self.bars, self._target(vals), **kw)


def normed(v: np.ndarray) -> np.ndarray:
    m = np.abs(v).max()
    return v / m if m > 1e-12 else v


# --------------------------------------------------------------------------
# the scene
# --------------------------------------------------------------------------


class EquitablePartitionStory(Scene):
    def construct(self):
        self.sec_network()
        self.sec_partition()
        self.sec_equitable()
        self.sec_lift()
        self.sec_invariance()
        self.sec_complement()
        self.sec_diagram()

    # -- 1. the network ----------------------------------------------------
    def sec_network(self):
        self.next_section("network")
        g = NeuronGraph()
        g.scale(0.82).move_to(DOWN * 0.45)
        self.graph = g

        title = Text("Equitable partitions", font_size=44).to_edge(UP, buff=0.4)
        sub = Text("8 neurons, 13 connections", font_size=26, color=GREY_A)
        sub.next_to(title, DOWN, buff=0.22)

        self.play(Write(title), run_time=1.2)
        self.play(LaggedStart(*[GrowFromCenter(g.dots[n]) for n in range(1, 9)], lag_ratio=0.08),
                  LaggedStart(*[FadeIn(g.labels[n]) for n in range(1, 9)], lag_ratio=0.08),
                  run_time=1.6)
        self.play(LaggedStart(*[Create(e) for e in g.edge_group], lag_ratio=0.06), run_time=2.0)
        self.play(FadeIn(sub, shift=DOWN * 0.15))
        self.wait(0.8)
        self.title, self.sub = title, sub

    # -- 2. the partition --------------------------------------------------
    def sec_partition(self):
        self.next_section("partition")
        g = self.graph
        new_sub = Text("colour every neuron by its role", font_size=26, color=GREY_A)
        new_sub.move_to(self.sub)
        self.play(FadeOut(self.sub, shift=UP * 0.15), FadeIn(new_sub, shift=UP * 0.15))
        self.sub = new_sub

        self.play(*[g.dots[n].animate.set_fill(NODE_COL[n], 1.0) for n in range(1, 9)],
                  run_time=1.2)

        hulls, hlabels = VGroup(), VGroup()
        for k, ns in CLUSTERS.items():
            box = SurroundingRectangle(
                VGroup(*[g.dots[n] for n in ns]), color=CCOL[k], corner_radius=0.35, buff=0.3
            ).set_stroke(width=3, opacity=0.85).set_z_index(-1)
            lab = Text(f"role {k + 1}", font_size=24, color=CCOL[k]).next_to(box, DOWN, buff=0.18)
            hulls.add(box)
            hlabels.add(lab)
        self.play(LaggedStart(*[Create(b) for b in hulls], lag_ratio=0.25),
                  LaggedStart(*[FadeIn(l) for l in hlabels], lag_ratio=0.25), run_time=1.8)
        self.wait(0.6)
        self.hulls, self.hlabels = hulls, hlabels

    # -- 3. why it is equitable -------------------------------------------
    def sec_equitable(self):
        self.next_section("equitable")
        g = self.graph
        left = VGroup(g, self.hulls, self.hlabels)

        new_sub = Text("count each neuron's neighbours, role by role",
                       font_size=26, color=GREY_A).move_to(self.sub)
        self.play(FadeOut(self.sub, shift=UP * 0.15), FadeIn(new_sub, shift=UP * 0.15))
        self.sub = new_sub

        self.play(left.animate.scale(0.72).to_edge(LEFT, buff=0.35).shift(DOWN * 0.25),
                  run_time=1.2)

        # divisor matrix, revealed one row at a time
        mat = Matrix([["1", "3", "0"], ["2", "0", "1"], ["0", "1", "2"]],
                     h_buff=1.0, v_buff=0.9).scale(0.85)
        mat.to_edge(RIGHT, buff=1.6).shift(DOWN * 0.35)
        for i, row in enumerate(mat.get_rows()):
            for j, ent in enumerate(row):
                ent.set_color(CCOL[j])
        col_lab = VGroup(*[Text(f"{j + 1}", font_size=22, color=CCOL[j]) for j in range(3)])
        for j, cl in enumerate(col_lab):
            cl.next_to(mat.get_columns()[j], UP, buff=0.3)
        row_lab = VGroup(*[Text(f"{i + 1}", font_size=22, color=CCOL[i]) for i in range(3)])
        for i, rl in enumerate(row_lab):
            rl.move_to([mat.get_left()[0] - 0.35, mat.get_rows()[i].get_center()[1], 0])
        name = MathTex("B", font_size=40).next_to(row_lab, LEFT, buff=0.45)
        cap = VGroup(
            MathTex(r"B_{ij}:\ \text{role-}j\text{ neighbours}", font_size=28, color=GREY_A),
            MathTex(r"\text{of a role-}i\text{ neuron}", font_size=28, color=GREY_A),
        ).arrange(DOWN, buff=0.22)
        cap.next_to(mat, DOWN, buff=0.55)

        rows = mat.get_rows()
        self.play(FadeIn(mat.get_brackets()), FadeIn(col_lab), FadeIn(row_lab),
                  Write(name), FadeIn(cap))
        for r in rows:
            r.set_opacity(0.0)

        self.tally_anchor = np.array([mat.get_center()[0], 1.95, 0.0])
        self.tally = VGroup()

        order = [(0, [1, 2]), (1, [3, 4, 5]), (2, [6, 7, 8])]
        for k, ns in order:
            for i, n in enumerate(ns):
                self._inspect(n, slow=(i == 0 and k <= 1))
            self.play(rows[k].animate.set_opacity(1.0),
                      Flash(rows[k], color=CCOL[k], line_length=0.15), run_time=0.9)
            self.wait(0.25)

        self._restore_edges()
        self.play(FadeOut(self.tally))
        verdict = Text("every neuron in a role sees the same number of\n"
                       "neurons in each role  ⇒  the partition is equitable",
                       font_size=24, line_spacing=0.9).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(verdict, shift=UP * 0.2))
        self.wait(1.6)
        self.play(FadeOut(verdict), FadeOut(cap))
        self.divisor = VGroup(mat, col_lab, row_lab, name)

    def _inspect(self, n: int, slow: bool = False):
        g = self.graph
        inc = g.incident(n)
        dim = [e.animate.set_stroke(GREY_D, 2, opacity=0.25)
               for e in g.edge_group if e not in inc]
        lit = [e.animate.set_stroke(WHITE, 5, opacity=1.0) for e in inc]
        rings = [g.dots[m].animate.set_stroke(YELLOW, 5) for m in NBRS[n]]
        plain = [g.dots[m].animate.set_stroke(WHITE, 2.5)
                 for m in range(1, 9) if m not in NBRS[n]]

        c = counts(n)
        new = VGroup(
            Text(f"neuron {n}", font_size=24),
            MathTex(f"({c[0]},\\,{c[1]},\\,{c[2]})", font_size=46),
        ).arrange(DOWN, buff=0.25)
        for j, part in enumerate([1, 3, 5]):
            new[1][0][part].set_color(CCOL[j])
        new.move_to(self.tally_anchor)

        rt = 1.0 if slow else 0.55
        anims = dim + lit + rings + plain
        if len(self.tally) == 0:
            self.tally.become(new)
            self.play(*anims, FadeIn(self.tally), run_time=rt)
        else:
            self.play(*anims, Transform(self.tally, new), run_time=rt)
        self.wait(0.5 if slow else 0.15)

    def _restore_edges(self):
        g = self.graph
        self.play(*[e.animate.set_stroke(EDGE_BASE, 3, opacity=0.7) for e in g.edge_group],
                  *[g.dots[m].animate.set_stroke(WHITE, 2.5) for m in range(1, 9)],
                  run_time=0.6)

    # -- 4. the lift x = H y ----------------------------------------------
    def sec_lift(self):
        self.next_section("lift")
        self.play(FadeOut(self.divisor), FadeOut(self.sub), FadeOut(self.title),
                  FadeOut(self.hlabels))
        left = VGroup(self.graph, self.hulls)
        self.play(left.animate.scale(0.62).to_edge(LEFT, buff=0.4).shift(UP * 0.2),
                  run_time=1.0)

        title = Text("one number per role, copied to every neuron in it",
                     font_size=30).to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=DOWN * 0.15))

        Hm = Matrix([["1", "0", "0"], ["1", "0", "0"],
                     ["0", "1", "0"], ["0", "1", "0"], ["0", "1", "0"],
                     ["0", "0", "1"], ["0", "0", "1"], ["0", "0", "1"]],
                    h_buff=0.8, v_buff=0.62).scale(0.62)
        yv = Matrix([["y_1"], ["y_2"], ["y_3"]], v_buff=0.75).scale(0.72)
        xv = Matrix([["y_1"], ["y_1"], ["y_2"], ["y_2"], ["y_2"], ["y_3"], ["y_3"], ["y_3"]],
                    v_buff=0.62).scale(0.62)

        for i, row in enumerate(Hm.get_rows()):
            row.set_color(CCOL[CLUSTER_OF[i + 1]])
        for k, ent in enumerate(yv.get_entries()):
            ent.set_color(CCOL[k])
        for i, ent in enumerate(xv.get_entries()):
            ent.set_color(CCOL[CLUSTER_OF[i + 1]])

        lhs = MathTex("x", "=", font_size=48)
        eq = VGroup(lhs, Hm, yv).arrange(RIGHT, buff=0.3)
        eq2 = VGroup(MathTex("=", font_size=48), xv).arrange(RIGHT, buff=0.3)
        whole = VGroup(eq, eq2).arrange(RIGHT, buff=0.45)
        whole.move_to([1.7, -0.25, 0])

        hlab = Text("H", font_size=24, color=GREY_A).next_to(Hm, DOWN, buff=0.25)
        ylab = Text("role state", font_size=20, color=GREY_A).next_to(yv, DOWN, buff=0.25)
        xlab = Text("neuron state", font_size=20, color=GREY_A).next_to(xv, DOWN, buff=0.25)

        self.play(Write(lhs), FadeIn(Hm), FadeIn(yv), FadeIn(hlab), FadeIn(ylab), run_time=1.4)
        self.play(FadeIn(eq2[0]), FadeIn(xv.get_brackets()), FadeIn(xlab))
        for ent in xv.get_entries():
            ent.set_opacity(0.0)

        for i in (0, 2, 5):
            r = SurroundingRectangle(Hm.get_rows()[i], color=YELLOW, buff=0.06)
            c = SurroundingRectangle(yv.get_entries(), color=YELLOW, buff=0.08)
            self.play(Create(r), Create(c), run_time=0.5)
            self.play(xv.get_entries()[i].animate.set_opacity(1.0), run_time=0.4)
            self.play(FadeOut(r), FadeOut(c), run_time=0.3)
        self.play(*[xv.get_entries()[i].animate.set_opacity(1.0)
                    for i in range(8)], run_time=0.7)

        note = Text("x = H y  —  the role subspace is everything H can reach",
                    font_size=26).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note, shift=UP * 0.15))
        self.wait(1.4)

        self.play(FadeOut(whole), FadeOut(hlab), FadeOut(ylab), FadeOut(xlab),
                  FadeOut(note), FadeOut(title))
        self.play(FadeOut(self.graph), FadeOut(self.hulls))

    # -- 5. invariance -----------------------------------------------------
    def sec_invariance(self):
        self.next_section("invariance")
        key = MathTex("A", "H", "=", "H", "B", font_size=64)
        key[1].set_color(YELLOW)
        key[3].set_color(YELLOW)
        self.play(Write(key), run_time=1.2)
        self.wait(0.6)
        conseq = MathTex(r"A\,(H y) \;=\; H\,(B y)", font_size=48).next_to(key, DOWN, buff=0.7)
        self.play(Write(conseq), run_time=1.2)
        gloss = Text("run the 8-neuron network on a lifted state and you get\n"
                     "another lifted state — the role subspace never leaks",
                     font_size=26, line_spacing=0.9, color=GREY_A)
        gloss.next_to(conseq, DOWN, buff=0.6)
        self.play(FadeIn(gloss, shift=UP * 0.15))
        self.wait(1.8)
        self.play(FadeOut(key), FadeOut(conseq), FadeOut(gloss))

        # side-by-side dynamics
        ypan = BarPanel([CCOL[k] for k in range(3)], width=0.55, gap=0.16, unit=1.25)
        ypan.move_to([-4.4, -1.05, 0])
        xpan = BarPanel(BAR_COL, width=0.45, gap=0.16, unit=1.25, group_sizes=[2, 3, 3])
        xpan.move_to([2.4, -1.05, 0])

        ylab = MathTex("y \\in \\mathbb{R}^3", font_size=34).move_to([-4.4, 1.45, 0])
        xlab = MathTex("x \\in \\mathbb{R}^8", font_size=34).move_to([2.4, 1.45, 0])
        ysub = Text("3 roles", font_size=20, color=GREY_A).move_to([-4.4, -2.75, 0])
        xsub = Text("8 neurons", font_size=20, color=GREY_A).move_to([2.4, -2.75, 0])

        arrow = Arrow([-3.0, -1.05, 0], [-0.55, -1.05, 0], buff=0.0, stroke_width=4,
                      color=YELLOW, max_tip_length_to_length_ratio=0.09)
        alab = MathTex("x = H y", font_size=34, color=YELLOW).next_to(arrow, UP, buff=0.25)

        step = VGroup(MathTex("y \\leftarrow B y", font_size=32).move_to([-4.4, 2.35, 0]),
                      MathTex("x \\leftarrow A x", font_size=32).move_to([2.4, 2.35, 0]))

        y = np.array([0.9, -0.3, 0.5])
        x = H @ y
        s = np.abs(x).max()
        y, x = y / s, x / s
        ypan.set_values(y)
        xpan.set_values(x)

        self.play(FadeIn(ypan), FadeIn(xpan), FadeIn(ylab), FadeIn(xlab),
                  FadeIn(ysub), FadeIn(xsub), GrowArrow(arrow), FadeIn(alab), run_time=1.2)
        self.play(FadeIn(step), run_time=0.6)

        for _ in range(6):
            y = normed(B @ y)
            x = normed(A @ x)
            self.play(ypan.anim(y), xpan.anim(x),
                      Indicate(step[0], scale_factor=1.15),
                      Indicate(step[1], scale_factor=1.15), run_time=0.85)
            self.wait(0.12)
        self.wait(0.4)

        check = Text("x stays flat inside every role, forever — the 8-dimensional\n"
                     "trajectory is just the 3-dimensional one, lifted",
                     font_size=25, line_spacing=0.9).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(check, shift=UP * 0.15))
        self.wait(2.0)
        self.play(FadeOut(VGroup(ypan, xpan, ylab, xlab, ysub, xsub, arrow, alab, step, check)))

    # -- 6. the complement -------------------------------------------------
    def sec_complement(self):
        self.next_section("complement")
        title = Text("what about a state that is not lifted?", font_size=32).to_edge(UP, buff=0.45)
        self.play(FadeIn(title, shift=DOWN * 0.15))

        pan = BarPanel(BAR_COL, unit=1.35, width=0.6, gap=0.24, group_sizes=[2, 3, 3])
        pan.move_to([0, -0.25, 0])

        rng = np.random.default_rng(3)
        P = np.eye(8) - H @ np.linalg.inv(H.T @ H) @ H.T
        x = normed(P @ rng.normal(size=8))
        pan.set_values(x)

        sums = VGroup(*[MathTex(r"\textstyle\sum = 0", font_size=32, color=CCOL[k])
                        for k in range(3)])
        groups = [[0, 1], [2, 3, 4], [5, 6, 7]]
        bx, by = pan.baseline.get_center()[0], pan.baseline.get_center()[1]
        for k, gidx in enumerate(groups):
            sums[k].move_to([bx + np.mean([pan.xs[i] for i in gidx]), by - 1.95, 0])

        self.play(FadeIn(pan), run_time=0.9)
        self.play(FadeIn(sums), run_time=0.7)
        note = Text("inside each role the values cancel: this state is orthogonal\n"
                    "to the role subspace",
                    font_size=25, line_spacing=0.9, color=GREY_A).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note))
        self.wait(1.4)

        step = MathTex("x \\leftarrow A x", font_size=34).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(step))
        for _ in range(7):
            x = normed(A @ x)
            self.play(pan.anim(x), Indicate(step, scale_factor=1.12), run_time=0.8)
            self.wait(0.1)

        punch = Text("the role sums stay exactly zero — this half of the state space\n"
                     "is invariant too, and never mixes into the roles",
                     font_size=25, line_spacing=0.9).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(note), FadeIn(punch, shift=UP * 0.15))
        self.wait(2.0)
        self.play(FadeOut(VGroup(pan, sums, step, punch, title)))

    # -- 7. the commuting square ------------------------------------------
    def sec_diagram(self):
        self.next_section("diagram")
        y0 = MathTex("y", font_size=54, color=YELLOW).move_to([-3.0, 1.5, 0])
        y1 = MathTex("B y", font_size=54, color=YELLOW).move_to([3.0, 1.5, 0])
        x0 = MathTex("H y", font_size=54).move_to([-3.0, -1.5, 0])
        x1 = MathTex("A H y", font_size=54).move_to([3.0, -1.5, 0])

        top = Arrow(y0.get_right(), y1.get_left(), buff=0.35, stroke_width=4)
        bot = Arrow(x0.get_right(), x1.get_left(), buff=0.35, stroke_width=4)
        lft = Arrow(y0.get_bottom(), x0.get_top(), buff=0.3, stroke_width=4, color=YELLOW)
        rgt = Arrow(y1.get_bottom(), x1.get_top(), buff=0.3, stroke_width=4, color=YELLOW)

        tl = MathTex("B", font_size=36).next_to(top, UP, buff=0.15)
        bl = MathTex("A", font_size=36).next_to(bot, DOWN, buff=0.15)
        ll = MathTex("H", font_size=36, color=YELLOW).next_to(lft, LEFT, buff=0.2)
        rl = MathTex("H", font_size=36, color=YELLOW).next_to(rgt, RIGHT, buff=0.2)

        rowl = Text("3 roles", font_size=22, color=GREY_A).next_to(y0, LEFT, buff=1.2)
        rowr = Text("8 neurons", font_size=22, color=GREY_A).next_to(x0, LEFT, buff=1.2)

        self.play(FadeIn(y0), FadeIn(rowl))
        self.play(GrowArrow(top), FadeIn(tl), FadeIn(y1))
        self.play(GrowArrow(lft), FadeIn(ll), FadeIn(x0), FadeIn(rowr))
        self.play(GrowArrow(bot), FadeIn(bl), FadeIn(x1))
        self.play(GrowArrow(rgt), FadeIn(rl))
        self.wait(0.6)

        ident = MathTex(r"A H y \;=\; H B y", font_size=44, color=YELLOW)
        ident.move_to([0.0, -2.5, 0])
        self.play(Write(ident), run_time=1.0)
        close = Text("lift then run  =  run then lift", font_size=32).to_edge(DOWN, buff=0.45)
        self.play(Write(close), run_time=1.2)
        self.wait(2.2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)
