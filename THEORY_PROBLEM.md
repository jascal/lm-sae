# Minimal-description / circuit-minimization of a trained language model

*A problem statement for outside input. Self-contained; notation defined below.*

## 1. Objects

Let $M$ be a trained autoregressive transformer. $M(\cdot \mid x_{1:t})$ is a distribution over a vocabulary
$\mathcal V$, $|\mathcal V| = V$. Residual width $d$, depth $L$, MLP hidden width $m \approx 4d$. Per layer (pre-LN):

$$ z = x + \mathrm{Attn}_\ell(\mathrm{LN}(x)), \qquad x' = z + \mathrm{MLP}_\ell(\mathrm{LN}(z)), \qquad
   \mathrm{MLP}_\ell(u) = W^{\downarrow}_\ell\,\sigma\!\big(W^{\uparrow}_\ell u\big), $$

with $\sigma=$ GELU, $W^{\uparrow}\!\in\mathbb R^{m\times d}$, $W^{\downarrow}\!\in\mathbb R^{d\times m}$. Let
$\mathcal D$ be the data distribution over contexts; activations are sampled $x\sim\mathcal D$.

**Known complexity.** For fixed depth and $O(\log n)$ precision, one forward pass computes a function in uniform
$\mathsf{TC}^0$ — constant-depth, polynomial-size **threshold circuits** (Merrill–Sabharwal 2023; Hao et al. on
saturated transformers). So a single step is a low-complexity, CPU-portable, *finite* arithmetic circuit; the
autoregressive loop with the context as scratchpad lifts the multi-step process toward $\mathsf P$ / Turing-complete.

> Consequence we keep front-of-mind: there is **no impossibility barrier** to running or simplifying $M$ — it is a finite
> $\mathsf{TC}^0$ circuit. The question is *constructive* (find a small/legible equivalent circuit), not a lower-bound
> question. And TC⁰ size lower bounds are famously open (we cannot even prove $\mathsf{TC}^0 \neq \mathsf{NP}$), so
> "irreducible" is not provable even in principle — only "we have not found a small decomposition."

## 2. The empirical phenomenon ("the forge tax")

Split $M$'s next-token behavior into **retrieval** (a bounded-context lookup reproduces the top-1) + **composition**
(the rest). Define the **decompilable fraction**

$$ \rho(M) \;:=\; \Pr_{x\sim\mathcal D}\Big[\hat M_{\mathrm{flat}}(x) = \textstyle\arg\max_v M(v\mid x)\Big], $$

where $\hat M_{\mathrm{flat}}$ is the best bounded-context predictor (n-gram back-off + in-context induction copy).

Measured facts (GPT-2 and Pythia 14M–1.4B):

1. $\rho \approx 0.5$ for GPT-2; **decreasing** with scale/capability, $0.56 \to 0.45$ across 14M→1.4B.
2. The non-retrieval part is **MLP-carried** (mean-ablating MLPs hurts "content" next-tokens far more than ablating
   attention; gap widens with scale) and **context-bound** (held-out NLL on content keeps falling out to $\geq 64$
   tokens of context).
3. It resists sparsification in every **static** basis tried — per-token *active dimension* stays $\Theta(m)$:
   - low-rank $W \approx AB$ in the activation norm $\lVert X(W-AB)\rVert_F$ needs rank $r \approx \tfrac{2}{3}d$ for
     near-lossless, robustly across {Frobenius / activation-weighted} × {no-retrain / trained} × {linear / GELU
     bottleneck} × {per-matrix / cross-layer Tucker / CP};
   - top-$k$ of the GELU hidden per token needs $k \approx 0.5\!-\!0.7\,m$ to recover content;
   - an $L_1$ sparse autoencoder up to $43\times d$ wide stays at $\sim\!1600$ active units (and uses *more* as it widens);
   - static expert clustering (MoEfication) needs $\sim\!0.5\!-\!0.75\,K$ of $K$ experts.
4. **Scaling.** The active fraction needed to recover the composition **grows** with $d$ ($\approx 48\% \to 68\%$ of
   $m$ from 70M→1B). The irreducible part is a *growing* fraction of the model.

## 3. The goal, and four precise sub-problems

**Overall goal.** Given the trained weights, produce the *minimal CPU program* — a flat lookup table $T$ plus an
arithmetic kernel $g$ — with $\mathbb E_{x\sim\mathcal D}\,\mathrm{KL}(M(\cdot\mid x)\,\Vert\,(T\!\oplus\!g)(x)) \le \epsilon$,
and characterize the size and structure of the irreducible kernel $g$ and how it scales with $d$.

**P1 — Data-restricted circuit minimization.** For $f(u)=W^{\downarrow}\sigma(W^{\uparrow}u)$ restricted to the
activation manifold $\mathrm{supp}(\mathcal D)$, characterize
$\;\mathcal C_\epsilon(f) = \min\{\,\mathrm{size}(g) : \mathbb E_{u\sim\mathcal D}\,\lVert g(u)-f(u)\rVert \le \epsilon\,\}$.
Is there a (possibly learned, overcomplete) basis in which $f|_{\mathcal D}$ has an $s$-sparse *per-input* evaluation
with $s=o(m)$? Evidence says **no** for all static / $L_1$ bases; the open lever is a *learned, input-conditional*
sparse evaluation (a trained router — currently under test).

**P2 — The lookup/compute boundary.** Write $M = M_{\mathrm{LUT}} \oplus M_{\mathrm{comp}}$, $M_{\mathrm{LUT}}$ the
part agreeing with a finite-context table. Bound the circuit complexity / description length of $M_{\mathrm{comp}}$.
Why is $M_{\mathrm{comp}}$ **not itself a (bigger) lookup** — i.e., what certifies it as genuine bounded-depth
*computation* rather than memorization? (Empirically, unbounded corpus context saturates near trigram; more context
does not recover $M_{\mathrm{comp}}$.)

**P3 — Scaling law of the irreducible fraction.** Under a superposition model of the MLP ($m' \gg m$ latent features
encoded in near-orthogonal directions among $m$ neurons), predict the per-token effective active dimension
$s(d,m,H)$ ($H$ = conditional entropy of the data) and explain why $s/m$ **grows** with scale.

**P4 — Conditional sparsity (the optimization we want).** Find a router $g_\theta:\mathbb R^d\to\{0,1\}^K$ and expert
partition of the $m$ neurons minimizing
$\;\mathbb E_x\,\mathrm{KL}\!\big(M(\cdot\mid x)\,\Vert\,\tilde M_\theta(\cdot\mid x)\big) + \lambda\,\mathbb E_x\lVert g_\theta(x)\rVert_0$,
where $\tilde M_\theta$ evaluates only the selected experts. Is $\inf_\theta$ active-count $\ll m$ at fixed KL? Static
routing gives no win; the *learned* version is the open question.

## 4. Pointers that would help

(i) data-restricted / average-case circuit complexity and MDL; (ii) superposition & sparse-coding capacity bounds
(Elhage et al. "Toy Models of Superposition"; compressed sensing) for P3; (iii) conditional-computation / learned MoE
routing theory and lower bounds for P4; (iv) whether $M_{\mathrm{comp}}$ lives in a natural subclass of $\mathsf{TC}^0$
(e.g. $\mathsf{ACC}^0$, depth-2 linear-threshold, $\mathrm{MAJ}\circ\mathrm{MAJ}$); (v) any handle on **average-case /
data-restricted** circuit *minimization* given the worst-case TC⁰ lower-bound barrier.
