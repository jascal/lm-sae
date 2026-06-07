---
title: GPT-2 (small) disassembly
---

# GPT-2 (small) — per-head disassembly

**GPT-2 / absolute-position.** GPT-2 disassembly (12 layers x 12 heads + MLP). first-order; operand basis nt=40. ADDR=where-to-read  WRITE=copy/transform (OV)  bind=top content binding (B_h)  role=circuit (shared mean-write 'default' direction omitted — see write_bus_check)

Operator roles referenced (hyperlinked inline below): [induction](../operators/induction.md) · [prevtok](../operators/prevtok.md) · [structural](../operators/structural.md). Full raw listing: [`gpt2_disassembly.txt`](https://github.com/jascal/lm-sae/blob/main/docs/listings/gpt2_disassembly.txt). See the [operator catalog](../operators/README.md) for what each role means.

_First-order, single-component reads (+ the induction idiom); provisional. Each head line: head · ADDR (where it reads) · WRITE (copy/transform) · top content binding · operator role. Lines like `L.MLP.n####` are **MLP neurons** (the COMPUTE class — `n####` is the neuron's index in that layer's gated-MLP intermediate dimension, e.g. Gemma-2-2B has 9216/layer), **not** attention heads; each lists the top read-tokens → write-tokens (the layer's most salient few)._


### Layer 0

- `L0.H0  ADDR=diffuse      WRITE=copy      bind "'d"->'_is'` [`induction`](../operators/induction.md)
- `L0.H1  ADDR=content      WRITE=transform bind '_is'->'_are'`
- `L0.H2  ADDR=diffuse      WRITE=transform bind ','->';'`
- `L0.H3  ADDR=content      WRITE=transform bind '_your'->'_you'`
- `L0.H4  ADDR=relative-Δ   WRITE=transform bind 'IA'->'MAR'`
- `L0.H5  ADDR=content      WRITE=transform bind '_he'->'_his'`
- `L0.H6  ADDR=diffuse      WRITE=copy      bind '_to'->'_for'`
- `L0.H7  ADDR=relative-Δ   WRITE=transform bind 'I'->'CI'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L0.H8  ADDR=structural   WRITE=transform bind '_you'->'_I'` [`line-anchor`](../operators/structural.md)
- `L0.H9  ADDR=structural   WRITE=transform bind '_for'->'_to'` [`line-anchor`](../operators/structural.md)
- `L0.H10 ADDR=structural   WRITE=transform bind '_your'->'_you'`
- `L0.H11 ADDR=structural   WRITE=transform bind '_the'->'_his'` [`line-anchor`](../operators/structural.md)
- `L0.MLP.n2733 reads {_of,_that,_is} -> writes {_the,_our,_to}`
- `L0.MLP.n1899 reads {_in,,,_I} -> writes {_the,_our,_his}`
- `L0.MLP.n1612 reads {_with,_in,_of} -> writes {,,_the,.}`

### Layer 1

- `L1.H0  ADDR=relative-Δ   WRITE=transform bind '_they'->'_you'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L1.H1  ADDR=structural   WRITE=copy      bind ','->';'` [`line-anchor`](../operators/structural.md)
- `L1.H2  ADDR=structural   WRITE=transform bind 'Ċ'->'First'`
- `L1.H3  ADDR=absolute-sink WRITE=transform bind '_that'->'_it'`
- `L1.H4  ADDR=structural   WRITE=transform bind '_that'->'_it'`
- `L1.H5  ADDR=structural   WRITE=transform bind '_they'->'_he'` [`line-anchor`](../operators/structural.md)
- `L1.H6  ADDR=structural   WRITE=transform bind 'Ċ'->':'`
- `L1.H7  ADDR=diffuse      WRITE=transform bind 'US'->'_the'`
- `L1.H8  ADDR=structural   WRITE=transform bind "'"->'_is'` [`line-anchor`](../operators/structural.md)
- `L1.H9  ADDR=diffuse      WRITE=transform bind 'US'->'_with'`
- `L1.H10 ADDR=structural   WRITE=transform bind '_not'->'_for'`
- `L1.H11 ADDR=content      WRITE=transform bind '_is'->'_are'`
- `L1.MLP.n1120 reads {_for,_to,And} -> writes {_for,First,_he}`
- `L1.MLP.n2401 reads {_me,_not,.} -> writes {_for,First,_he}`
- `L1.MLP.n242  reads {.,_have,!} -> writes {_for,First,And}`

### Layer 2

- `L2.H0  ADDR=relative-Δ   WRITE=transform bind ','->'Ċ'`
- `L2.H1  ADDR=diffuse      WRITE=copy      bind 'IA'->'_the'`
- `L2.H2  ADDR=relative-Δ   WRITE=transform bind '_to'->'_for'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L2.H3  ADDR=structural   WRITE=transform bind '_is'->'_to'` `[prev-tok→induction-feed,line-anchor]`
- `L2.H4  ADDR=relative-Δ   WRITE=transform bind '_that'->'_it'`
- `L2.H5  ADDR=relative-Δ   WRITE=transform bind ';'->'_for'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L2.H6  ADDR=structural   WRITE=transform bind '_his'->'_you'` [`line-anchor`](../operators/structural.md)
- `L2.H7  ADDR=structural   WRITE=transform bind '_for'->'_of'`
- `L2.H8  ADDR=relative-Δ   WRITE=transform bind '_is'->'_for'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L2.H9  ADDR=relative-Δ   WRITE=transform bind 'Ċ'->'First'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L2.H10 ADDR=diffuse      WRITE=transform bind 'Ċ'->'First'`
- `L2.H11 ADDR=absolute-sink WRITE=transform bind 'Ċ'->'First'`
- `L2.MLP.n3034 reads {And,_he,_of} -> writes {_for,First,And}`
- `L2.MLP.n666  reads {First,And,_for} -> writes {_for,_are,_is}`
- `L2.MLP.n1825 reads {And,_for,_he} -> writes {.,_the,'d}`

### Layer 3

- `L3.H0  ADDR=content      WRITE=copy      bind '_is'->'_are'`
- `L3.H1  ADDR=structural   WRITE=transform bind '_to'->'_it'` [`line-anchor`](../operators/structural.md)
- `L3.H2  ADDR=relative-Δ   WRITE=transform bind '_of'->'_the'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L3.H3  ADDR=relative-Δ   WRITE=transform bind '_that'->'_him'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L3.H4  ADDR=absolute-sink WRITE=transform bind '_they'->'_you'`
- `L3.H5  ADDR=diffuse      WRITE=transform bind '_your'->'_they'`
- `L3.H6  ADDR=relative-Δ   WRITE=transform bind 'Ċ'->'IA'`
- `L3.H7  ADDR=relative-Δ   WRITE=transform bind ':'->'IA'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L3.H8  ADDR=relative-Δ   WRITE=transform bind '_of'->'_his'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L3.H9  ADDR=relative-Δ   WRITE=transform bind '.'->'Ċ'`
- `L3.H10 ADDR=absolute-sink WRITE=transform bind 'US'->'IA'`
- `L3.H11 ADDR=structural   WRITE=transform bind '_our'->'_you'` [`line-anchor`](../operators/structural.md)
- `L3.MLP.n1751 reads {_for,And,First} -> writes {First,_for,And}`
- `L3.MLP.n1027 reads {_to,_for,MAR} -> writes {'d,And,First}`
- `L3.MLP.n1874 reads {_to,_and,_his} -> writes {And,First,_he}`

### Layer 4

- `L4.H0  ADDR=relative-Δ   WRITE=transform bind 'I'->'MAR'`
- `L4.H1  ADDR=relative-Δ   WRITE=transform bind ':'->'Ċ'`
- `L4.H2  ADDR=structural   WRITE=transform bind '_he'->'_I'`
- `L4.H3  ADDR=relative-Δ   WRITE=copy      bind '_he'->'US'`
- `L4.H4  ADDR=structural   WRITE=transform bind '_is'->'_are'`
- `L4.H5  ADDR=structural   WRITE=transform bind '.'->'Ċ'`
- `L4.H6  ADDR=diffuse      WRITE=transform bind '_our'->'_you'`
- `L4.H7  ADDR=structural   WRITE=transform bind 'MAR'->'First'`
- `L4.H8  ADDR=absolute-sink WRITE=transform bind 'US'->'IA'`
- `L4.H9  ADDR=diffuse      WRITE=transform bind ':'->'.'`
- `L4.H10 ADDR=absolute-sink WRITE=transform bind 'IA'->'US'`
- `L4.H11 ADDR=relative-Δ   WRITE=transform bind ':'->'IA'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L4.MLP.n923  reads {_for,First,_to} -> writes {_for,First,And}`
- `L4.MLP.n541  reads {.,?,!} -> writes {.,!,?}`
- `L4.MLP.n462  reads {.,:,?} -> writes {.,!,:}`

### Layer 5

- `L5.H0  ADDR=absolute-sink WRITE=transform bind 'MAR'->'IA'` [`induction`](../operators/induction.md)
- `L5.H1  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'` [`induction`](../operators/induction.md)
- `L5.H2  ADDR=relative-Δ   WRITE=transform bind 'MAR'->':'`
- `L5.H3  ADDR=structural   WRITE=transform bind ':'->'Ċ'`
- `L5.H4  ADDR=relative-Δ   WRITE=transform bind 'CI'->':'`
- `L5.H5  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'` [`induction`](../operators/induction.md)
- `L5.H6  ADDR=relative-Δ   WRITE=copy      bind 'CI'->'MAR'`
- `L5.H7  ADDR=absolute-sink WRITE=transform bind '_with'->"'d"`
- `L5.H8  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'` [`induction`](../operators/induction.md)
- `L5.H9  ADDR=absolute-sink WRITE=transform bind 'IA'->':'`
- `L5.H10 ADDR=diffuse      WRITE=transform bind '_I'->'_me'`
- `L5.H11 ADDR=absolute-sink WRITE=transform bind 'US'->'IA'`
- `L5.MLP.n877  reads {.,?,!} -> writes {.,?,!}`
- `L5.MLP.n1821 reads {.,Ċ,MAR} -> writes {.,?,!}`
- `L5.MLP.n1661 reads {_of,_with,_to} -> writes {_of,_in,_with}`

### Layer 6

- `L6.H0  ADDR=relative-Δ   WRITE=copy      bind 'MAR'->'Ċ'`
- `L6.H1  ADDR=absolute-sink WRITE=transform bind '.'->'Ċ'`
- `L6.H2  ADDR=absolute-sink WRITE=transform bind '_for'->'_your'`
- `L6.H3  ADDR=absolute-sink WRITE=transform bind 'MAR'->':'`
- `L6.H4  ADDR=diffuse      WRITE=transform bind 'MAR'->'IA'`
- `L6.H5  ADDR=absolute-sink WRITE=transform bind '.'->'MAR'`
- `L6.H6  ADDR=absolute-sink WRITE=transform bind 'US'->'IA'`
- `L6.H7  ADDR=diffuse      WRITE=transform bind 'MAR'->'First'`
- `L6.H8  ADDR=relative-Δ   WRITE=copy      bind 'CI'->'MAR'` [`prev-tok→induction-feed`](../operators/prevtok.md)
- `L6.H9  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'` [`induction`](../operators/induction.md)
- `L6.H10 ADDR=absolute-sink WRITE=transform bind 'MAR'->'IA'`
- `L6.H11 ADDR=relative-Δ   WRITE=transform bind 'MAR'->'IA'`
- `L6.MLP.n2432 reads {_I,_me,I} -> writes {_I,_me,I}`
- `L6.MLP.n1256 reads {.,?,!} -> writes {.,?,!}`
- `L6.MLP.n1443 reads {,,I,_with} -> writes {,,;,_and}`

### Layer 7

- `L7.H0  ADDR=relative-Δ   WRITE=copy      bind 'CI'->':'`
- `L7.H1  ADDR=absolute-sink WRITE=copy      bind 'MAR'->'CI'`
- `L7.H2  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L7.H3  ADDR=absolute-sink WRITE=transform bind 'MAR'->'IA'`
- `L7.H4  ADDR=absolute-sink WRITE=copy      bind '_and'->'_have'`
- `L7.H5  ADDR=absolute-sink WRITE=transform bind 'MAR'->'IA'`
- `L7.H6  ADDR=absolute-sink WRITE=transform bind 'MAR'->'US'`
- `L7.H7  ADDR=absolute-sink WRITE=transform bind 'MAR'->'IA'`
- `L7.H8  ADDR=relative-Δ   WRITE=copy      bind 'US'->'CI'`
- `L7.H9  ADDR=absolute-sink WRITE=transform bind '.'->'IA'`
- `L7.H10 ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L7.H11 ADDR=absolute-sink WRITE=transform bind 'US'->':'` [`induction`](../operators/induction.md)
- `L7.MLP.n658  reads {_a,Ċ,?} -> writes {_a,_it,_him}`
- `L7.MLP.n2985 reads {_I,_me,I} -> writes {_I,I,_me}`
- `L7.MLP.n2474 reads {_of,_in,_with} -> writes {_of,_in,_with}`

### Layer 8

- `L8.H0  ADDR=structural   WRITE=transform bind 'And'->'_a'` [`line-anchor`](../operators/structural.md)
- `L8.H1  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L8.H2  ADDR=structural   WRITE=copy      bind 'MAR'->'IA'`
- `L8.H3  ADDR=absolute-sink WRITE=transform bind 'MAR'->'US'`
- `L8.H4  ADDR=absolute-sink WRITE=copy      bind '.'->'Ċ'`
- `L8.H5  ADDR=diffuse      WRITE=copy      bind 'US'->'MAR'`
- `L8.H6  ADDR=absolute-sink WRITE=transform bind 'US'->':'`
- `L8.H7  ADDR=relative-Δ   WRITE=copy      bind '_our'->'_of'`
- `L8.H8  ADDR=diffuse      WRITE=transform bind 'MAR'->'IA'`
- `L8.H9  ADDR=absolute-sink WRITE=transform bind 'US'->':'`
- `L8.H10 ADDR=absolute-sink WRITE=transform bind 'MAR'->'US'`
- `L8.H11 ADDR=absolute-sink WRITE=transform bind 'I'->'US'`
- `L8.MLP.n1333 reads {_a,_to,:} -> writes {_a,_I,_is}`
- `L8.MLP.n157  reads {_I,_me,I} -> writes {_I,_me,I}`
- `L8.MLP.n201  reads {_have,_with,'s} -> writes {_have,_with,'s}`

### Layer 9

- `L9.H0  ADDR=absolute-sink WRITE=copy      bind 'US'->':'`
- `L9.H1  ADDR=absolute-sink WRITE=transform bind 'MAR'->'IA'`
- `L9.H2  ADDR=absolute-sink WRITE=transform bind 'MAR'->'IA'`
- `L9.H3  ADDR=relative-Δ   WRITE=copy      bind 'I'->'MAR'`
- `L9.H4  ADDR=absolute-sink WRITE=transform bind 'US'->':'`
- `L9.H5  ADDR=absolute-sink WRITE=transform bind '_me'->'.'` [`induction`](../operators/induction.md)
- `L9.H6  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L9.H7  ADDR=absolute-sink WRITE=transform bind 'And'->'_you'`
- `L9.H8  ADDR=absolute-sink WRITE=copy      bind '_it'->'_of'`
- `L9.H9  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L9.H10 ADDR=absolute-sink WRITE=copy      bind '_our'->'_of'`
- `L9.H11 ADDR=structural   WRITE=transform bind 'US'->':'`
- `L9.MLP.n1240 reads {_a,_me,_in} -> writes {_a,_is,_are}`
- `L9.MLP.n376  reads {_his,_your,_our} -> writes {_his,_your,_our}`
- `L9.MLP.n2859 reads {.,?,!} -> writes {.,?,!}`

### Layer 10

- `L10.H0  ADDR=absolute-sink WRITE=copy      bind 'MAR'->'CI'`
- `L10.H1  ADDR=absolute-sink WRITE=transform bind '_they'->'_have'`
- `L10.H2  ADDR=absolute-sink WRITE=copy      bind 'US'->':'`
- `L10.H3  ADDR=absolute-sink WRITE=transform bind 'MAR'->'IA'`
- `L10.H4  ADDR=structural   WRITE=copy      bind '.'->'Ċ'`
- `L10.H5  ADDR=absolute-sink WRITE=copy      bind 'And'->'First'`
- `L10.H6  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L10.H7  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L10.H8  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L10.H9  ADDR=absolute-sink WRITE=copy      bind '_your'->'_for'`
- `L10.H10 ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L10.H11 ADDR=absolute-sink WRITE=copy      bind '_our'->'_of'`
- `L10.MLP.n772  reads {_a,Ċ,!} -> writes {_a,_your,_the}`
- `L10.MLP.n2114 reads {Ċ,'s,_that} -> writes {Ċ,?,:}`
- `L10.MLP.n2288 reads {_have,'s,Ċ} -> writes {_have,'s,_is}`

### Layer 11

- `L11.H0  ADDR=structural   WRITE=copy      bind '!'->'.'` `[induction,line-anchor]`
- `L11.H1  ADDR=absolute-sink WRITE=copy      bind '_he'->'_to'`
- `L11.H2  ADDR=absolute-sink WRITE=copy      bind 'MAR'->'CI'`
- `L11.H3  ADDR=diffuse      WRITE=copy      bind '_of'->'_the'`
- `L11.H4  ADDR=diffuse      WRITE=transform bind 'MAR'->'CI'`
- `L11.H5  ADDR=absolute-sink WRITE=copy      bind 'And'->'_I'`
- `L11.H6  ADDR=absolute-sink WRITE=transform bind 'MAR'->'CI'`
- `L11.H7  ADDR=absolute-sink WRITE=copy      bind '_and'->'_your'`
- `L11.H8  ADDR=structural   WRITE=transform bind 'US'->'CI'`
- `L11.H9  ADDR=absolute-sink WRITE=copy      bind 'MAR'->'CI'`
- `L11.H10 ADDR=diffuse      WRITE=copy      bind 'MAR'->'CI'`
- `L11.H11 ADDR=structural   WRITE=copy      bind 'And'->'_he'`
- `L11.MLP.n2679 reads {Ċ,.,MAR} -> writes {.,Ċ,:}`
- `L11.MLP.n1550 reads {_I,_they,I} -> writes {_I,_they,_you}`
- `L11.MLP.n43   reads {_a,;,_not} -> writes {_a,_for,_with}`

_Generated from the committed listing by `disassembly_pages.py`._