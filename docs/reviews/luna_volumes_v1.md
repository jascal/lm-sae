# Review of *Buke de lore fira* and the field volumes

I read the core book first, then `logic.md`, `physics.md`, `philosophy.md`, `morality.md`, `digital.md`, and `mathematics.md` in full. The files were accessible. I learned the head-before-dependent tree notation, `pe`/`te` relators, and the dependent-count endings from the core. I could parse many field sentences, but the volumes often introduce roots through context before their dictionary entry appears; the translations below therefore distinguish confidence.

## Can the volumes be read after the core?

Yes, in outline. Definitions, examples, formulas, and repeated sentence patterns make the subjects recognizable. No, not reliably enough for a future model to treat every sentence as unambiguous prose: several technical roots are only inferable, and the volumes do not consistently distinguish a definition, quotation, assertion, and exercise.

### Logic (`logic.md`)

1. “`bi buke de gogika pe tisa .`” — “This is the logic book for the reader.” **Confidence: medium.** `buke` and `gogika` are explicit; the function of `tisa` is inferred from repeated book introductions.
2. “`bi rira pe bi hoba pe bofe la .`” — “The sun is true.” **Confidence: low.** I can identify `rira` as true and `bofe` as sun, but the nesting and `hoba` are not defined clearly enough to know whether this is a proposition or an example object.
3. “`bi falasa pe ferase tisa .`” — “The sentence is false.” **Confidence: high.** This is the explicit liar-paradox setup in section **si videka lemite de gogika**.

### Physics (`physics.md`)

1. “`bi metare 1.8-a pe lenite de Ana-a .`” — “Ana’s length is 1.8 metres.” **Confidence: high.** The local dictionary defines metre and measurement.
2. “`bi mapori m-a a-a pe F-a .`” — “F equals m times a.” **Confidence: high.** The displayed tree in **si mosina fesa** makes the formula unambiguous.
3. “`bi metari 299792458-a ne sekude pona pe sidede de luta .`” — “Light has a speed of 299,792,458 per second.” **Confidence: medium-high.** The intended unit is metres per second, but the sentence does not explicitly repeat `metar` in the compound.

### Philosophy (`philosophy.md`)

1. “`gini ne vudera pe filofa .`” — “Philosophy begins with wonder.” **Confidence: medium-high.** `gini`, `vudera`, and `filofa` are clear from the dictionary and heading.
2. “`desi te sevi te nada pe ma pe Socrates-a .`” — “Socrates says that he knows nothing.” **Confidence: medium.** The nested `sevi`/`nada` construction is readable, but the quotation boundary is not marked.
3. “`donami pe Zhuangzi-a kase bi terila pe ha .`” — “Zhuangzi dreams that he is a butterfly.” **Confidence: medium-high.** This is the familiar butterfly-dream story, though `terila` is not separately defined in the volume.

### Morality (`morality.md`)

1. “`musi te dare vasa pe vora .`” — “We must ask a question.” **Confidence: medium.** The modal and question roots are clear, but the subject is supplied by context rather than marked in this sentence.
2. “`desi te rule pona pe radine vana .`” — “The first rule is about the many.” **Confidence: low-medium.** `rule`, `pona`, and `vana` are identifiable; `radine` is not defined locally, so the exact English relation is uncertain.
3. “`raki te bi vasa pe ruli guta ka pe dulita .`” — “Ask whether the rule is good as a duty.” **Confidence: medium-high.** The tree diagram immediately above fixes the attachment.

### Digital (`digital.md`)

1. “`livi ne vule tova pe geneta .`” — “Two agents live in the world.” **Confidence: medium.** The number and `geneta` are clear, but the existential construction is learned only from examples.
2. “`saheni te sekude nake kepola pe time unix-a .`” — “Count the seconds since the Unix epoch.” **Confidence: high.** The section **time unix-a** defines all relevant roots.
3. “`runi te manede git_status-a pe ma .`” — “Run the `git_status` command on me.” **Confidence: high.** The literal command is clearly a loan and the surrounding shell dialogue supplies the role.

### Mathematics (`mathematics.md`)

1. “`bi fura pe si tova tova .`” — “Four is two plus two.” **Confidence: high.** The displayed tree and arithmetic section make this the intended reading.
2. “`bi mapori pi-a povi r-a tova pe mare de sirula .`” — “The area of a circle is pi times the square of its radius.” **Confidence: medium-high.** The geometry roots are explicit; `povi r-a tova` is the power construction.
3. “`pefe bi masume de pirime tova pe mero vevena keda marore gahe tova .`” — “Every even number greater than two is the sum of two primes.” **Confidence: high.** This is the Goldbach conjecture statement in **vetune de Goldbach-a**.

## Problems and misleading forms

### [unclear] Quotation versus assertion

The core teaches that a sentence is a tree, with the head preceding dependents, and introduces arguments through relators such as `pe` and `te`. It does not give a quotation convention. Thus in `philosophy.md`, section **vudera**, “`desi te sevi te nada pe ma pe Socrates-a .`” could be a narrator’s assertion or a quotation attributed to Socrates. Add a quotation marker and a convention for speaker/source attribution.

### [unclear] Technical roots that are too close

The dictionaries contain easily confusable pairs:

> `morat-u nomuna morality-a Moral-a moral-a`  
> `morol-u detiva mortal-a sterblich-a mortal-a`

The first pair is especially dangerous in a moral text: one vowel changes “morality” to “mortal.” Give a pronunciation/orthography warning or choose a less confusable root.

### [content] Digital volume gives destructive commands without operational qualification

In section **sonele la** it says:

> `runi te rm_-rf_/-a nera .`

This is a valid example of a shell command’s shape, but presented without saying that it recursively deletes the filesystem. A future agent could copy it as an instruction. Label it explicitly as dangerous and add a harmless dry-run example.

### [content] Physics needs status labels for claims

In **si safa nivera**:

> `bi neba pe mabisa . sere bo pare 13800000000-a veta pe nivere la .`

The age of the universe is presented as an unqualified fact. The number is a reasonable approximate age, but the book gives no unit, uncertainty, or “approximately” marker. Add a measurement convention and mark estimates as estimates.

### [content] Mathematical conjecture is not separated from theorem

The Goldbach sentence is:

> `pefe bi masume de pirime tova pe mero vevena keda marore gahe tova .`

The heading calls it a conjecture, but the prose gives no explicit epistemic marker distinguishing “conjectured” from “proved.” Add a standard `conjecture` frame and a `proved` frame; otherwise a model may learn it as established mathematics.

### [suggestion] Attribution needs sources

The philosophy and morality volumes name Socrates, Heraclitus, Plato, Kant, Descartes, Hume, Zhuangzi, Wittgenstein, Confucius, Aristotle, Bentham, Rawls, and others, but supply no source or quotation status. For example:

> `deso te tisa nora pe Heraclitus-a .`

Add “traditional paraphrase,” “direct quotation,” or “author’s summary,” plus a source-language citation. This prevents misattributing paraphrases as exact quotations.

## What the volumes need first

They need a short per-volume grammar key; explicit labels for definition, example, quotation, theorem, conjecture, and empirical observation; units and uncertainty notation in physics; proof/status conventions in mathematics; citation conventions in philosophy and morality; and safety annotations for executable digital literals. They also need a root index that lists every root before its first use, and one English-free worked parse for each new construction.

## A passage that works well

The mathematics presentation of four is exemplary:

> `bi`  
> `    fura`  
> `    pe`  
> `        si`  
> `            tova`  
> `            tova`

followed by “`bi fura pe si tova tova .`” in **mamaka**. The diagram, compact sentence, and dictionary roots align: a reader can verify both the Talema tree and the arithmetic without English scaffolding. That pattern should be reused for every new technical rule.
