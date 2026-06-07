---
title: Qwen-2.5-1.5B disassembly
---

# Qwen-2.5-1.5B — per-head disassembly

**RoPE / GQA / RMSNorm.** addr=where-to-read (attn bucket)  WRITE=copy/transform (OV diag)  bind=top QK token binding  idioms=behavioral role  QK/OV[...]=SAE-feature opcode (SAE layer only)

Operator roles referenced (hyperlinked inline below): [duplicate](../operators/duplicate.md) · [induction](../operators/induction.md) · [prevtok](../operators/prevtok.md). Full raw listing: [`qwen25_15b_disassembly.txt`](https://github.com/jascal/lm-sae/blob/main/docs/listings/qwen25_15b_disassembly.txt). See the [operator catalog](../operators/README.md) for what each role means.

> **Discovery pass (causal overlay).** The ★ badges below are from the cross-model discovery sweep ([discovered components](../operators/discovered.md), 3-seed): every head/MLP mean-ablated and ranked by its **induction-NLL** damage (base induction NLL 0.45). A head is flagged **⚠ UNNAMED-candidate** when it is load-bearing but matches no catalogued operator — a lead to dossier. **2 unnamed load-bearing** here: `1.6`, `1.5`. Only the sweep's top-ranked components carry a badge (most heads are not individually load-bearing).

_First-order, single-component reads (+ the induction idiom); provisional. Each head line: head · ADDR (where it reads) · WRITE (copy/transform) · top content binding · operator role · ★ discovery-pass causal (when load-bearing). Lines like `L.MLP.n####` are **MLP neurons** (the COMPUTE class — `n####` is the neuron's index in that layer's gated-MLP intermediate dimension, e.g. Gemma-2-2B has 9216/layer), **not** attention heads; each lists the top read-tokens → write-tokens (the layer's most salient few)._

- `Qwen/Qwen2.5-1.5B DISASSEMBLY  (28 layers x 12 heads + gated MLP; GQA n_kv=2, RoPE, RMSNorm)`
- `corpus=wikitext  tokens=10000  token-operands=40  SAE-feature opcode: n/a (token-operand basis only)`
- `attention budget (mean per-head mass): self 11%  sink 42%  prev 7%  structural 11%  local 13%  long_range 16%`
- `plumbing 84% | content (long-range) 16%`
- `causally load-bearing (induction-NLL ablation): 2.3, 14.0, 14.3, 14.4, 19.2, 19.3`

### Layer 0

- `0.0  addr=structural  WRITE=transform bind '_of'->'_by'`
- `0.1  addr=self        WRITE=transform bind '6'->'_the'`
- `0.2  addr=prev        WRITE=transform bind '>'->'_<'` [`prev-token`](../operators/prevtok.md)
- `0.3  addr=self        WRITE=transform bind '_to'->'_a'`
- `0.4  addr=long_range  WRITE=transform bind '_Ċ'->'>'`
- `0.5  addr=self        WRITE=transform bind '>'->'_Ċ'` [`prev-token`](../operators/prevtok.md)
- `0.6  addr=self        WRITE=transform bind '_was'->'_were'` — ★ causal: induction ΔNLL **+0.21**±0.02, 47% of base, generic +0.43
- `0.7  addr=self        WRITE=transform bind 'y'->'_Little'`
- `0.8  addr=prev        WRITE=transform bind '_Ċ'->'>'` [`prev-token`](../operators/prevtok.md)
- `0.9  addr=long_range  WRITE=transform bind '>'->'_<'` [`duplicate`](../operators/duplicate.md)
- `0.10  addr=self        WRITE=transform bind '_for'->'_of'` — ★ causal: induction ΔNLL **+0.50**±0.09, 110% of base, generic +0.35
- `0.11  addr=local       WRITE=transform bind 'y'->'_Little'`
- `0.MLP.n3970  reads {s,_Chronicles,_in} -> writes {s,y,_were}` — ★ causal: induction ΔNLL **+7.71**±0.11, 1703% of base, generic +3.05
- `0.MLP.n2981  reads {s,_Chronicles,y} -> writes {s,y,_were}`
- `0.MLP.n2693  reads {_The,unk,>} -> writes {_at,_=,0}`

### Layer 1

- `1.0  addr=local       WRITE=transform bind '0'->'_Rock'`
- `1.1  addr=local       WRITE=transform bind '_of'->'_"'` [`prev-token`](../operators/prevtok.md)
- `1.2  addr=long_range  WRITE=transform bind '_.'->'_Ċ'`
- `1.3  addr=long_range  WRITE=transform bind '_Ċ'->'_as'`
- `1.4  addr=prev        WRITE=transform bind '_game'->'_for'` [`prev-token`](../operators/prevtok.md) — ★ causal: induction ΔNLL **+0.19**±0.01, 42% of base, generic -0.00
- `1.5  addr=long_range  WRITE=transform bind '2'->'_'` — ★ causal: induction ΔNLL **+0.11**±0.02, 25% of base, generic +0.06 ⚠ [**UNNAMED-candidate**](../operators/discovered_xmodel.md)
- `1.6  addr=long_range  WRITE=transform bind "_'"->'_"'` — ★ causal: induction ΔNLL **+0.22**±0.08, 49% of base, generic +0.06 ⚠ [**UNNAMED-candidate**](../operators/discovered_xmodel.md)
- `1.7  addr=long_range  WRITE=transform bind '2'->'1'` [`duplicate`](../operators/duplicate.md)
- `1.8  addr=long_range  WRITE=transform bind '_.'->'_Ċ'`
- `1.9  addr=prev        WRITE=transform bind '_.'->'_"'` [`prev-token`](../operators/prevtok.md)
- `1.10  addr=self        WRITE=transform bind '_<'->'>'`
- `1.11  addr=local       WRITE=transform bind '_Ċ'->'_='`
- `1.MLP.n2754  reads {_and,_Chronicles,_Little} -> writes {_and,_,,_at}` — ★ causal: induction ΔNLL **+13.38**±0.08, 2956% of base, generic +8.03
- `1.MLP.n1162  reads {8,>,_Chronicles} -> writes {8,unk,9}`
- `1.MLP.n5601  reads {_=,_The,_.} -> writes {_=,_.,_The}`

### Layer 2

- `2.0  addr=sink        WRITE=transform bind '_.'->'_='`
- `2.1  addr=sink        WRITE=transform bind "_'"->'_"'`
- `2.2  addr=sink        WRITE=transform bind '_='->'>'`
- `2.3  addr=sink        WRITE=transform bind '_Valk'->'y'` [`induction`](../operators/induction.md) — ★ causal: induction ΔNLL **+0.45**±0.04, 99% of base, generic -0.00
- `2.4  addr=sink        WRITE=transform bind '_.'->'ria'`
- `2.5  addr=sink        WRITE=transform bind '_='->'>'` [`induction`](../operators/induction.md)
- `2.6  addr=sink        WRITE=transform bind '_.'->'_,'`
- `2.7  addr=sink        WRITE=transform bind '_,'->'_<'`
- `2.8  addr=self        WRITE=transform bind '_'->'0'`
- `2.9  addr=sink        WRITE=transform bind '_"'->'_<'`
- `2.10  addr=sink        WRITE=transform bind '_"'->'>'`
- `2.11  addr=sink        WRITE=transform bind 'ria'->'y'`
- `2.MLP.n2417  reads {_Rock,_Little,_Valk} -> writes {_Ċ,_@,s}` — ★ causal: induction ΔNLL **+13.75**±0.19, 3038% of base, generic +5.72
- `2.MLP.n3029  reads {_Rock,_=,>} -> writes {s,_@,_at}`
- `2.MLP.n5714  reads {_",_Rock,_Little} -> writes {s,_Ċ,_@}`

### Layer 3

- `3.0  addr=sink        WRITE=transform bind '_the'->'_Valk'`
- `3.1  addr=sink        WRITE=transform bind '_,'->'_'`
- `3.2  addr=sink        WRITE=transform bind 'y'->'>'` [`prev-token`](../operators/prevtok.md)
- `3.3  addr=sink        WRITE=transform bind '0'->'_'`
- `3.4  addr=sink        WRITE=transform bind "_'"->'_"'` [`duplicate`](../operators/duplicate.md)
- `3.5  addr=sink        WRITE=copy      bind '_Little'->'_.'`
- `3.6  addr=sink        WRITE=transform bind '>'->'_Valk'`
- `3.7  addr=long_range  WRITE=transform bind '_.'->'_"'`
- `3.8  addr=sink        WRITE=transform bind '_.'->"_'"`
- `3.9  addr=sink        WRITE=transform bind '_<'->'_.'`
- `3.10  addr=sink        WRITE=transform bind '_.'->'_,'`
- `3.11  addr=structural  WRITE=transform bind '_"'->'_,'`
- `3.MLP.n4157  reads {_<,_Little,_in} -> writes {_Ċ,s,_@}`
- `3.MLP.n4941  reads {_",_<,_Rock} -> writes {_=,_as,9}`
- `3.MLP.n3626  reads {_of,_to,0} -> writes {>,_The,_by}`

### Layer 4

- `4.0  addr=sink        WRITE=transform bind '_.'->'_,'`
- `4.1  addr=sink        WRITE=transform bind '_='->'_<'`
- `4.2  addr=sink        WRITE=copy      bind '2'->'_Valk'`
- `4.3  addr=sink        WRITE=copy      bind '_='->'>'`
- `4.4  addr=sink        WRITE=transform bind '_<'->'>'`
- `4.5  addr=sink        WRITE=transform bind '_='->'_Ċ'`
- `4.6  addr=sink        WRITE=transform bind '_.'->'_Ċ'`
- `4.7  addr=sink        WRITE=transform bind '_a'->'_was'`
- `4.8  addr=sink        WRITE=transform bind '0'->"_'"`
- `4.9  addr=sink        WRITE=transform bind '_the'->'_with'`
- `4.10  addr=sink        WRITE=copy      bind '_"'->"_'"` [`duplicate`](../operators/duplicate.md)
- `4.11  addr=sink        WRITE=transform bind '_were'->'_of'`
- `4.MLP.n610   reads {_Little,_Rock,>} -> writes {_Ċ,s,_@}`
- `4.MLP.n4228  reads {_of,_Chronicles,0} -> writes {>,_<,1}`
- `4.MLP.n3430  reads {_to,0,ria} -> writes {ria,_to,0}`

### Layer 5

- `5.0  addr=sink        WRITE=transform bind '_Ċ'->'>'`
- `5.1  addr=self        WRITE=transform bind '_a'->'_with'`
- `5.2  addr=sink        WRITE=copy      bind '_a'->'_as'`
- `5.3  addr=sink        WRITE=transform bind '_a'->'_Valk'`
- `5.4  addr=long_range  WRITE=copy      bind '_<'->'_.'`
- `5.5  addr=sink        WRITE=copy      bind '_<'->'_,'`
- `5.6  addr=sink        WRITE=transform bind '6'->'_@'`
- `5.7  addr=sink        WRITE=transform bind '_'->'0'`
- `5.8  addr=sink        WRITE=copy      bind '_a'->'_was'`
- `5.9  addr=sink        WRITE=transform bind '_'->'0'` [`duplicate`](../operators/duplicate.md)
- `5.10  addr=sink        WRITE=transform bind '_The'->'_='` [`duplicate`](../operators/duplicate.md)
- `5.11  addr=sink        WRITE=transform bind '_<'->'>'` [`duplicate`](../operators/duplicate.md)
- `5.MLP.n8729  reads {_Rock,_in,>} -> writes {_and,s,_The}`
- `5.MLP.n5209  reads {_the,unk,_<} -> writes {_the,>,_"}`
- `5.MLP.n5242  reads {_the,_",_'} -> writes {_a,_Valk,_Ċ}`

### Layer 6

- `6.0  addr=long_range  WRITE=copy      bind '_Ċ'->'s'`
- `6.1  addr=sink        WRITE=copy      bind '_<'->'>'` [`duplicate`](../operators/duplicate.md)
- `6.2  addr=structural  WRITE=transform bind '_The'->'>'`
- `6.3  addr=sink        WRITE=copy      bind 's'->'_were'`
- `6.4  addr=sink        WRITE=transform bind '_were'->'_with'`
- `6.5  addr=sink        WRITE=copy      bind '_,'->'>'`
- `6.6  addr=sink        WRITE=transform bind '_The'->'_='` [`duplicate`](../operators/duplicate.md)
- `6.7  addr=sink        WRITE=transform bind '_in'->'unk'` [`duplicate`](../operators/duplicate.md)
- `6.8  addr=sink        WRITE=transform bind '>'->'_,'`
- `6.9  addr=sink        WRITE=transform bind 'ria'->'_,'`
- `6.10  addr=sink        WRITE=transform bind '2'->'_,'`
- `6.11  addr=sink        WRITE=copy      bind 's'->'_,'`
- `6.MLP.n1293  reads {_,_the,_.} -> writes {_,_<,_'}`
- `6.MLP.n146   reads {_of,_in,_game} -> writes {_of,_game,_in}`
- `6.MLP.n7085  reads {_the,_,_to} -> writes {_the,_,_Chronicles}`

### Layer 7

- `7.0  addr=sink        WRITE=transform bind '0'->'_'`
- `7.1  addr=sink        WRITE=transform bind '_<'->'_,'`
- `7.2  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `7.3  addr=self        WRITE=transform bind '_,'->'_'`
- `7.4  addr=long_range  WRITE=transform bind '>'->'_"'`
- `7.5  addr=sink        WRITE=copy      bind '_Rock'->'_was'`
- `7.6  addr=sink        WRITE=transform bind '_.'->'_were'`
- `7.7  addr=local       WRITE=transform bind '_.'->'_Ċ'`
- `7.8  addr=prev        WRITE=transform bind 'y'->'_were'` [`prev-token`](../operators/prevtok.md)
- `7.9  addr=local       WRITE=transform bind '_.'->'_Ċ'`
- `7.10  addr=local       WRITE=transform bind '_,'->'2'`
- `7.11  addr=long_range  WRITE=transform bind 'ria'->'_were'`
- `7.MLP.n2034  reads {_,_the,_,} -> writes {_,_,,_the}`
- `7.MLP.n5514  reads {_,_,,y} -> writes {_,_the,>}`
- `7.MLP.n8492  reads {_,0,6} -> writes {_,0,_to}`

### Layer 8

- `8.0  addr=sink        WRITE=transform bind '_Ċ'->'_"'`
- `8.1  addr=sink        WRITE=transform bind '_"'->'>'` [`duplicate`](../operators/duplicate.md)
- `8.2  addr=long_range  WRITE=transform bind 's'->'_Chronicles'`
- `8.3  addr=sink        WRITE=transform bind '_'->'_,'` [`duplicate`](../operators/duplicate.md) — ★ causal: induction ΔNLL **+0.65**±0.01, 143% of base, generic +0.00
- `8.4  addr=sink        WRITE=copy      bind '9'->'0'` [`duplicate`](../operators/duplicate.md)
- `8.5  addr=sink        WRITE=copy      bind '_='->'_<'`
- `8.6  addr=sink        WRITE=copy      bind '_,'->'_by'`
- `8.7  addr=sink        WRITE=copy      bind '_'->'0'` [`induction`](../operators/induction.md)
- `8.8  addr=sink        WRITE=transform bind '_='->'>'`
- `8.9  addr=sink        WRITE=transform bind '_The'->'s'`
- `8.10  addr=sink        WRITE=transform bind '_the'->'_with'`
- `8.11  addr=sink        WRITE=copy      bind '_,'->'_by'`
- `8.MLP.n4582  reads {_the,_to,_'} -> writes {_the,_,,_}`
- `8.MLP.n8067  reads {_=,_of,2} -> writes {_of,_=,_The}`
- `8.MLP.n408   reads {_to,y,_in} -> writes {_to,_.,_,}`

### Layer 9

- `9.0  addr=self        WRITE=transform bind '_were'->'_and'`
- `9.1  addr=sink        WRITE=transform bind '_<'->'>'`
- `9.2  addr=sink        WRITE=copy      bind '0'->'_'`
- `9.3  addr=sink        WRITE=transform bind '_<'->'_"'`
- `9.4  addr=sink        WRITE=transform bind '_.'->'_Ċ'`
- `9.5  addr=sink        WRITE=copy      bind '_and'->'_='`
- `9.6  addr=sink        WRITE=transform bind '_Rock'->'_Chronicles'`
- `9.7  addr=sink        WRITE=transform bind 'unk'->'>'` [`duplicate`](../operators/duplicate.md)
- `9.8  addr=sink        WRITE=copy      bind 'unk'->'>'`
- `9.9  addr=sink        WRITE=transform bind '_.'->'>'`
- `9.10  addr=sink        WRITE=transform bind '_Valk'->'y'`
- `9.11  addr=sink        WRITE=copy      bind '_The'->'_@'`
- `9.MLP.n2615  reads {_,_,,_"} -> writes {_,0,_to}`
- `9.MLP.n3320  reads {_Ċ,_Valk,_Little} -> writes {_Ċ,_was,s}`
- `9.MLP.n6419  reads {_to,_the,_in} -> writes {_to,_the,_in}`

### Layer 10

- `10.0  addr=sink        WRITE=transform bind '_='->'_<'`
- `10.1  addr=sink        WRITE=copy      bind 's'->'_were'`
- `10.2  addr=long_range  WRITE=transform bind '_Ċ'->'s'`
- `10.3  addr=sink        WRITE=copy      bind '_,'->'_by'`
- `10.4  addr=sink        WRITE=copy      bind '_<'->'unk'`
- `10.5  addr=self        WRITE=transform bind '_Valk'->'y'`
- `10.6  addr=sink        WRITE=copy      bind '_Rock'->'_a'`
- `10.7  addr=sink        WRITE=copy      bind '0'->'_'`
- `10.8  addr=sink        WRITE=transform bind '_were'->'_by'`
- `10.9  addr=sink        WRITE=transform bind '_<'->'unk'`
- `10.10  addr=sink        WRITE=copy      bind '0'->'_'`
- `10.11  addr=sink        WRITE=copy      bind '_<'->'_='`
- `10.MLP.n8202  reads {_the,_to,_in} -> writes {_the,_,,_}`
- `10.MLP.n3312  reads {_,_,,_"} -> writes {_<,_",_}`
- `10.MLP.n8780  reads {_the,unk,_"} -> writes {_the,_,,_.}`

### Layer 11

- `11.0  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `11.1  addr=sink        WRITE=transform bind '_.'->'>'`
- `11.2  addr=local       WRITE=transform bind '_,'->'_Ċ'`
- `11.3  addr=sink        WRITE=transform bind '_The'->'_and'`
- `11.4  addr=prev        WRITE=transform bind '_were'->'_"'` [`prev-token`](../operators/prevtok.md)
- `11.5  addr=sink        WRITE=transform bind '_<'->'_='`
- `11.6  addr=structural  WRITE=transform bind '_<'->'unk'`
- `11.7  addr=long_range  WRITE=transform bind 'unk'->'_<'`
- `11.8  addr=self        WRITE=transform bind '_The'->'9'`
- `11.9  addr=sink        WRITE=copy      bind 'unk'->'_<'`
- `11.10  addr=long_range  WRITE=copy      bind '_<'->'unk'`
- `11.11  addr=sink        WRITE=transform bind '_Valk'->'_as'`
- `11.MLP.n3540  reads {_=,_in,_game} -> writes {_=,_game,_of}`
- `11.MLP.n614   reads {_in,_=,_of} -> writes {_=,_of,_in}`
- `11.MLP.n7472  reads {_,_to,0} -> writes {0,_.,_=}`

### Layer 12

- `12.0  addr=sink        WRITE=copy      bind '0'->'_'`
- `12.1  addr=sink        WRITE=copy      bind '_'->'0'` [`duplicate`](../operators/duplicate.md)
- `12.2  addr=sink        WRITE=transform bind '_Ċ'->'s'`
- `12.3  addr=sink        WRITE=transform bind '0'->'_'`
- `12.4  addr=sink        WRITE=transform bind '_<'->'>'`
- `12.5  addr=sink        WRITE=transform bind '_.'->'_Ċ'`
- `12.6  addr=sink        WRITE=copy      bind '_'->'0'`
- `12.7  addr=sink        WRITE=transform bind '_were'->'_Ċ'`
- `12.8  addr=sink        WRITE=copy      bind '_='->'unk'`
- `12.9  addr=sink        WRITE=copy      bind '_'->'0'`
- `12.10  addr=sink        WRITE=transform bind '_'->'0'`
- `12.11  addr=sink        WRITE=transform bind '_'->'0'` [`duplicate`](../operators/duplicate.md)
- `12.MLP.n2510  reads {_,_<,_the} -> writes {_,_the,_<}`
- `12.MLP.n7767  reads {_the,_,_to} -> writes {_the,y,_'}`
- `12.MLP.n7717  reads {_,,_,_"} -> writes {_",_,,_.}`

### Layer 13

- `13.0  addr=local       WRITE=transform bind '_'->'0'`
- `13.1  addr=sink        WRITE=transform bind '_'->'0'`
- `13.2  addr=local       WRITE=transform bind '_,'->'8'`
- `13.3  addr=sink        WRITE=transform bind '_<'->"_'"`
- `13.4  addr=prev        WRITE=transform bind 'y'->'_Rock'` [`prev-token`](../operators/prevtok.md) — ★ causal: induction ΔNLL **+0.25**±0.02, 54% of base, generic +0.01
- `13.5  addr=sink        WRITE=copy      bind '0'->'unk'`
- `13.6  addr=local       WRITE=transform bind 'unk'->'_<'`
- `13.7  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `13.8  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `13.9  addr=sink        WRITE=transform bind 'unk'->'_'`
- `13.10  addr=sink        WRITE=transform bind '_='->'_<'`
- `13.11  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `13.MLP.n8618  reads {_the,_,_to} -> writes {_the,_,unk}`
- `13.MLP.n1001  reads {_,unk,_the} -> writes {_,_the,_to}`
- `13.MLP.n3034  reads {_,0,_the} -> writes {_,0,8}`

### Layer 14

- `14.0  addr=sink        WRITE=transform bind '_'->'0'` [`induction`](../operators/induction.md)
- `14.1  addr=sink        WRITE=copy      bind '_'->'0'` [`induction`](../operators/induction.md)
- `14.2  addr=sink        WRITE=transform bind '_'->'0'` [`induction`](../operators/induction.md)
- `14.3  addr=sink        WRITE=transform bind '_'->'0'` [`induction`](../operators/induction.md)
- `14.4  addr=sink        WRITE=copy      bind '_'->'0'` [`induction`](../operators/induction.md)
- `14.5  addr=sink        WRITE=copy      bind '_'->'0'`
- `14.6  addr=sink        WRITE=transform bind '_'->'0'`
- `14.7  addr=sink        WRITE=transform bind '>'->'unk'`
- `14.8  addr=local       WRITE=transform bind '_.'->'_Ċ'`
- `14.9  addr=self        WRITE=transform bind '_,'->'_.'`
- `14.10  addr=sink        WRITE=transform bind '_='->'>'`
- `14.11  addr=sink        WRITE=transform bind '_.'->'_Ċ'`
- `14.MLP.n6811  reads {_,_<,_"} -> writes {_,_to,_in}`
- `14.MLP.n8187  reads {_,0,_.} -> writes {_,0,8}`
- `14.MLP.n5283  reads {_,,_,_.} -> writes {_,_,,_.}`

### Layer 15

- `15.0  addr=sink        WRITE=copy      bind 'unk'->'_<'`
- `15.1  addr=long_range  WRITE=transform bind 'unk'->'_<'`
- `15.2  addr=sink        WRITE=transform bind '_<'->'_='`
- `15.3  addr=sink        WRITE=transform bind '_'->'0'`
- `15.4  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `15.5  addr=self        WRITE=transform bind '_Valk'->'y'`
- `15.6  addr=structural  WRITE=transform bind 'unk'->'_<'`
- `15.7  addr=self        WRITE=transform bind '_that'->'_as'`
- `15.8  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `15.9  addr=sink        WRITE=copy      bind '2'->'_='`
- `15.10  addr=sink        WRITE=transform bind '_.'->'_and'`
- `15.11  addr=local       WRITE=transform bind 'y'->'_the'`
- `15.MLP.n4458  reads {_,unk,0} -> writes {_,unk,9}`
- `15.MLP.n5675  reads {_,,_.,_to} -> writes {_,,_.,_"}`
- `15.MLP.n6444  reads {_,0,y} -> writes {_,0,9}`

### Layer 16

- `16.0  addr=sink        WRITE=copy      bind '_'->'0'`
- `16.1  addr=sink        WRITE=transform bind '_'->'0'`
- `16.2  addr=sink        WRITE=copy      bind '_'->'0'`
- `16.3  addr=sink        WRITE=transform bind '_'->'0'`
- `16.4  addr=sink        WRITE=transform bind '_'->'0'`
- `16.5  addr=sink        WRITE=copy      bind 'unk'->'_<'`
- `16.6  addr=sink        WRITE=transform bind '_'->'0'`
- `16.7  addr=sink        WRITE=transform bind '_<'->'unk'`
- `16.8  addr=local       WRITE=copy      bind '_game'->'_with'`
- `16.9  addr=structural  WRITE=transform bind '_='->'>'`
- `16.10  addr=sink        WRITE=transform bind '_'->'0'`
- `16.11  addr=sink        WRITE=transform bind '_'->'0'`
- `16.MLP.n352   reads {_,,_,_the} -> writes {_,,_.,_the}`
- `16.MLP.n7143  reads {_,,_,unk} -> writes {_,,_.,_and}`
- `16.MLP.n8042  reads {_,0,9} -> writes {_,0,8}`

### Layer 17

- `17.0  addr=local       WRITE=transform bind 'unk'->'_<'`
- `17.1  addr=sink        WRITE=transform bind '_<'->'unk'`
- `17.2  addr=sink        WRITE=transform bind 'unk'->'_<'` [`duplicate`](../operators/duplicate.md)
- `17.3  addr=sink        WRITE=transform bind '_Rock'->'_Chronicles'`
- `17.4  addr=local       WRITE=transform bind 'y'->'_Valk'`
- `17.5  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `17.6  addr=sink        WRITE=transform bind '_'->'0'`
- `17.7  addr=sink        WRITE=transform bind '_='->'>'`
- `17.8  addr=sink        WRITE=copy      bind '_'->'0'`
- `17.9  addr=sink        WRITE=transform bind '_='->'_"'`
- `17.10  addr=sink        WRITE=transform bind '_'->'0'`
- `17.11  addr=sink        WRITE=transform bind '_'->'0'`
- `17.MLP.n3835  reads {_the,_,,_to} -> writes {_the,_.,_,}`
- `17.MLP.n1356  reads {_,,_,_"} -> writes {_.,_,,_}`
- `17.MLP.n4508  reads {_the,_to,_,} -> writes {_the,_.,_}`

### Layer 18

- `18.0  addr=sink        WRITE=transform bind 'unk'->'_'`
- `18.1  addr=sink        WRITE=transform bind '_'->'0'`
- `18.2  addr=local       WRITE=transform bind '_.'->'_"'` [`prev-token`](../operators/prevtok.md)
- `18.3  addr=sink        WRITE=transform bind '_'->'0'`
- `18.4  addr=sink        WRITE=transform bind 'y'->'_the'`
- `18.5  addr=sink        WRITE=transform bind '_<'->'unk'`
- `18.6  addr=sink        WRITE=transform bind '_<'->'unk'`
- `18.7  addr=sink        WRITE=transform bind '_'->'0'`
- `18.8  addr=prev        WRITE=transform bind '_and'->'_='` [`prev-token`](../operators/prevtok.md) — ★ causal: induction ΔNLL **+0.24**±0.03, 54% of base, generic +0.01
- `18.9  addr=sink        WRITE=transform bind '_<'->'unk'`
- `18.10  addr=self        WRITE=transform bind '_to'->'_the'`
- `18.11  addr=local       WRITE=copy      bind '_Ċ'->"_'"`
- `18.MLP.n4150  reads {_,_the,_.} -> writes {_,_<,_the}` — ★ causal: induction ΔNLL **+0.22**±0.01, 48% of base, generic +0.03
- `18.MLP.n658   reads {_the,_The,ria} -> writes {_the,_The,_"}`
- `18.MLP.n6280  reads {_,,_.,_"} -> writes {_,,_.,_to}`

### Layer 19

- `19.0  addr=sink        WRITE=transform bind '_'->'0'`
- `19.1  addr=sink        WRITE=transform bind '_'->'0'`
- `19.2  addr=sink        WRITE=transform bind '_<'->'unk'` [`induction`](../operators/induction.md)
- `19.3  addr=sink        WRITE=transform bind '_<'->'unk'` [`induction`](../operators/induction.md)
- `19.4  addr=sink        WRITE=transform bind '_'->'0'`
- `19.5  addr=sink        WRITE=transform bind '_<'->'unk'` [`induction`](../operators/induction.md)
- `19.6  addr=self        WRITE=transform bind '_='->'_Ċ'`
- `19.7  addr=sink        WRITE=transform bind '0'->'_'`
- `19.8  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `19.9  addr=long_range  WRITE=copy      bind '_='->"_'"`
- `19.10  addr=sink        WRITE=transform bind '_Ċ'->'_were'`
- `19.11  addr=local       WRITE=copy      bind '_.'->'_Ċ'` [`prev-token`](../operators/prevtok.md)
- `19.MLP.n2523  reads {_the,_The,_in} -> writes {_the,_The,_<}`
- `19.MLP.n6669  reads {_,0,_<} -> writes {_,_<,_the}`
- `19.MLP.n748   reads {_,_<,_'} -> writes {_,_<,_"}`

### Layer 20

- `20.0  addr=sink        WRITE=transform bind '_'->'0'`
- `20.1  addr=sink        WRITE=transform bind '_'->'0'`
- `20.2  addr=sink        WRITE=transform bind '_'->'0'`
- `20.3  addr=sink        WRITE=copy      bind '_'->'0'`
- `20.4  addr=sink        WRITE=transform bind '_'->'0'`
- `20.5  addr=sink        WRITE=transform bind '_'->'0'`
- `20.6  addr=sink        WRITE=copy      bind '_<'->'unk'`
- `20.7  addr=self        WRITE=transform bind '>'->'_='`
- `20.8  addr=local       WRITE=copy      bind '_Valk'->"_'"`
- `20.9  addr=sink        WRITE=copy      bind '_'->'0'`
- `20.10  addr=sink        WRITE=transform bind '_'->'0'`
- `20.11  addr=sink        WRITE=copy      bind '_<'->'_='`
- `20.MLP.n4274  reads {_,y,unk} -> writes {_,unk,_<}`
- `20.MLP.n2619  reads {_the,_to,_The} -> writes {_the,_The,_to}`
- `20.MLP.n1551  reads {_,_<,_'} -> writes {_,_,,_was}`

### Layer 21

- `21.0  addr=sink        WRITE=transform bind '_of'->'ria'`
- `21.1  addr=sink        WRITE=transform bind '9'->'_Chronicles'`
- `21.2  addr=sink        WRITE=copy      bind '_<'->'unk'`
- `21.3  addr=sink        WRITE=copy      bind '1'->'_'`
- `21.4  addr=sink        WRITE=copy      bind '2'->'s'`
- `21.5  addr=long_range  WRITE=transform bind '_<'->'_='`
- `21.6  addr=sink        WRITE=transform bind '_'->'0'`
- `21.7  addr=sink        WRITE=copy      bind '_<'->'unk'`
- `21.8  addr=sink        WRITE=copy      bind '_'->'0'`
- `21.9  addr=sink        WRITE=transform bind '_'->'0'`
- `21.10  addr=sink        WRITE=transform bind '_'->'0'`
- `21.11  addr=sink        WRITE=transform bind '_'->'0'`
- `21.MLP.n7614  reads {_the,_The,_to} -> writes {_the,_The,_to}`
- `21.MLP.n3680  reads {_the,_The,_} -> writes {_the,_The,_"}`
- `21.MLP.n6348  reads {_the,_The,_"} -> writes {_the,_a,unk}`

### Layer 22

- `22.0  addr=sink        WRITE=copy      bind '_the'->'_of'` [`prev-token`](../operators/prevtok.md)
- `22.1  addr=sink        WRITE=copy      bind '_the'->'_as'`
- `22.2  addr=sink        WRITE=copy      bind '_for'->'s'` [`prev-token`](../operators/prevtok.md)
- `22.3  addr=sink        WRITE=copy      bind '>'->'_by'`
- `22.4  addr=sink        WRITE=copy      bind '_The'->'_the'` [`duplicate`](../operators/duplicate.md)
- `22.5  addr=sink        WRITE=transform bind '_by'->'_for'`
- `22.6  addr=sink        WRITE=transform bind '_of'->'_Rock'`
- `22.7  addr=sink        WRITE=transform bind '_'->'0'`
- `22.8  addr=sink        WRITE=copy      bind '_'->'0'`
- `22.9  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `22.10  addr=sink        WRITE=transform bind '_'->'0'`
- `22.11  addr=sink        WRITE=transform bind '_'->'0'`
- `22.MLP.n3918  reads {_the,_The,_in} -> writes {_the,_The,_}`
- `22.MLP.n839   reads {_,unk,9} -> writes {_,_in,_Valk}`
- `22.MLP.n2277  reads {_the,_to,_in} -> writes {_the,_to,_.}`

### Layer 23

- `23.0  addr=sink        WRITE=transform bind '_<'->'unk'`
- `23.1  addr=sink        WRITE=copy      bind '_<'->'unk'` [`induction`](../operators/induction.md)
- `23.2  addr=sink        WRITE=transform bind '_<'->'unk'` [`induction`](../operators/induction.md)
- `23.3  addr=sink        WRITE=transform bind '_'->'0'`
- `23.4  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `23.5  addr=sink        WRITE=copy      bind '_of'->'_Rock'`
- `23.6  addr=sink        WRITE=transform bind '_the'->'_with'`
- `23.7  addr=sink        WRITE=transform bind '_'->'9'`
- `23.8  addr=sink        WRITE=transform bind '_<'->'unk'` [`induction`](../operators/induction.md)
- `23.9  addr=sink        WRITE=copy      bind '_a'->'_and'` [`prev-token`](../operators/prevtok.md)
- `23.10  addr=sink        WRITE=transform bind '_<'->'2'`
- `23.11  addr=sink        WRITE=copy      bind '_of'->'_in'`
- `23.MLP.n6145  reads {_the,_The,_to} -> writes {_the,_The,_to}`
- `23.MLP.n5893  reads {_to,_the,_,} -> writes {_to,_in,_of}`
- `23.MLP.n8360  reads {_in,_the,_'} -> writes {_in,_,,_the}`

### Layer 24

- `24.0  addr=sink        WRITE=transform bind '_and'->'_were'`
- `24.1  addr=sink        WRITE=transform bind '_a'->'_with'`
- `24.2  addr=sink        WRITE=transform bind '_='->'_Ċ'`
- `24.3  addr=sink        WRITE=transform bind '_The'->'_a'`
- `24.4  addr=sink        WRITE=copy      bind '_='->'_Ċ'`
- `24.5  addr=sink        WRITE=transform bind 'unk'->'_<'`
- `24.6  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `24.7  addr=sink        WRITE=transform bind 'unk'->'_"'`
- `24.8  addr=sink        WRITE=transform bind '_'->'0'`
- `24.9  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `24.10  addr=sink        WRITE=transform bind '_'->'0'`
- `24.11  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `24.MLP.n714   reads {s,_Ċ,_@} -> writes {_Ċ,_with,_as}`
- `24.MLP.n974   reads {_to,_the,_,} -> writes {_,,_in,_with}`
- `24.MLP.n6480  reads {_.,_",_} -> writes {_.,_",_the}`

### Layer 25

- `25.0  addr=sink        WRITE=transform bind '_'->'0'`
- `25.1  addr=sink        WRITE=copy      bind '1'->'_'`
- `25.2  addr=sink        WRITE=copy      bind '1'->'_'`
- `25.3  addr=self        WRITE=transform bind '_,'->'_and'`
- `25.4  addr=sink        WRITE=transform bind '_'->'0'`
- `25.5  addr=sink        WRITE=transform bind '1'->'0'`
- `25.6  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `25.7  addr=structural  WRITE=transform bind '>'->'_='`
- `25.8  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `25.9  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `25.10  addr=structural  WRITE=transform bind '_'->'0'`
- `25.11  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `25.MLP.n2015  reads {_the,_The,y} -> writes {_the,_The,_of}` — ★ causal: induction ΔNLL **+0.23**±0.01, 51% of base, generic +0.12
- `25.MLP.n8511  reads {_,9,unk} -> writes {_,_Little,_'}`
- `25.MLP.n2     reads {_",_,,_} -> writes {_,,_.,_"}`

### Layer 26

- `26.0  addr=sink        WRITE=transform bind '_'->'2'`
- `26.1  addr=sink        WRITE=copy      bind '_Ċ'->'_='`
- `26.2  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `26.3  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `26.4  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `26.5  addr=sink        WRITE=copy      bind '_at'->'_Rock'`
- `26.6  addr=sink        WRITE=transform bind '_and'->'_for'`
- `26.7  addr=sink        WRITE=transform bind '2'->'_'`
- `26.8  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `26.9  addr=sink        WRITE=copy      bind '0'->'_'`
- `26.10  addr=sink        WRITE=transform bind '2'->'_'`
- `26.11  addr=sink        WRITE=transform bind '_Ċ'->'_='`
- `26.MLP.n2908  reads {0,_',9} -> writes {s,_Ċ,_@}` — ★ causal: induction ΔNLL **+1.05**±0.02, 232% of base, generic +0.73
- `26.MLP.n7035  reads {0,_of,9} -> writes {s,_Ċ,_@}`
- `26.MLP.n974   reads {6,2,s} -> writes {s,_Ċ,_@}`

### Layer 27

- `27.0  addr=long_range  WRITE=copy      bind '_'->'2'`
- `27.1  addr=structural  WRITE=copy      bind '_<'->'unk'`
- `27.2  addr=long_range  WRITE=copy      bind '1'->'0'`
- `27.3  addr=long_range  WRITE=copy      bind '1'->'0'`
- `27.4  addr=structural  WRITE=transform bind '1'->'0'`
- `27.5  addr=long_range  WRITE=transform bind '1'->'0'`
- `27.6  addr=long_range  WRITE=copy      bind '9'->'0'`
- `27.7  addr=long_range  WRITE=transform bind '9'->'0'`
- `27.8  addr=long_range  WRITE=transform bind '9'->'2'`
- `27.9  addr=long_range  WRITE=copy      bind '_'->'2'`
- `27.10  addr=self        WRITE=transform bind '2'->'0'`
- `27.11  addr=long_range  WRITE=copy      bind '9'->'0'`
- `27.MLP.n7132  reads {_@,_',_=} -> writes {_.,_",_=}` — ★ causal: induction ΔNLL **+0.27**±0.05, 59% of base, generic +0.59
- `27.MLP.n2573  reads {_in,_of,_at} -> writes {_in,_of,9}`
- `27.MLP.n8467  reads {_,unk,_=} -> writes {_<,_Ċ,_'}`

_Generated from the committed listing + discovery sweep by `disassembly_pages.py`._