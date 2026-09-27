# Field chapter 3 · Logic. Łukasiewicz invented the head-first notation for logic (1924); Talema writes every
# sentence that way, so a connective is a relator head and its ending counts the statements it joins.
## logic

(speak (OBJ logic) already (SUBJ talem))                                     # Talema already speaks logic.
(be tree (SUBJ (sentence every)))                                            # Every sentence is a tree,
(say (OBJ (reason its)) (SUBJ (tree every)))                                 # and every tree says its reasons.
(write (OBJ logic) (with (head first/ADJ)) before (SUBJ "Łukasiewicz"))      # Łukasiewicz wrote logic with the head first.
(say (OBJ (whether (be true (SUBJ (statement the))))) (SUBJ logic))          # Logic says whether a statement is true,
(say (OBJ (whether (follow (from (statement other)) (SUBJ it)))) (SUBJ it))  # and whether it follows from other statements.

## (and true false)
(be (or true false) (SUBJ (statement every)))                                # Every statement is true or false.
(be statement (SUBJ (sentence (that (say (OBJ (thing a)) (about (world the)))))))   # A sentence that says a thing about the world is a statement.
(be statement not (SUBJ command))                                            # A command is not a statement,
(be statement not (SUBJ question))                                           # and a question is not a statement.
@verse
(be true (SUBJ (be hot (SUBJ (sun the)))))                                   # "The sun is hot" is true.
(be false (SUBJ (be cold (SUBJ (sun the)))))                                 # "The sun is cold" is false.
@prose
(be true (in (world every)) (SUBJ (statement necessary)))                   # A necessary statement is true in every world.
(be true (in (world some)) (SUBJ (statement possible)))                      # A possible statement is true in some world.
(be true (in (world no)) (SUBJ (statement impossible)))                      # An impossible statement is true in no world.
(mean (OBJ possible) (SUBJ (word "pol")))                                    # The word pol means possible.
(mean (OBJ (possible not)) (SUBJ (word "posib")))                            # The word posib means not possible: impossible.
(read (OBJ they) slowly)                                                     # Read them slowly.

(have (OBJ (mark three)) (SUBJ talem))                                       # Talema has three marks:
@verse
(prove (OBJ (statement the)) (SUBJ we))                                      # bov: we proved the statement,
(see (OBJ it) (SUBJ we))                                                     # ser: we saw it,
(know (OBJ (whether (be true (SUBJ it)))) not yet (SUBJ we))                 # pef: we do not yet know whether it is true.
@prose
(be (mark (of logic)) (SUBJ (mark the)))                                     # The marks are marks of logic.
(say (OBJ (thing (that (know (SUBJ we))))) (SUBJ they))                      # They say what we know.

## connective
(link/VERB (OBJ statement) (SUBJ connective))                                     # A connective joins statements.
(be head (SUBJ connective))                                                  # A connective is a head;
(be statement (SUBJ (child its)))                                            # its children are statements.
@verse
(have (OBJ (child one/NUM)) (SUBJ negation))                                 # Negation has one child: not.
(have (OBJ (child many)) (SUBJ conjunction))                                 # Conjunction has many children: and.
(have (OBJ (child many)) (SUBJ disjunction))                                 # Disjunction has many children: or.
(have (OBJ (child two)) (SUBJ implication))                                  # Implication has two children: if.
@prose
(be (word (of negation)) (SUBJ (word "f")))                                  # f is the word of negation,
(be (word (of conjunction)) (SUBJ (word "s")))                               # s is the word of conjunction,
(be (word (of disjunction)) (SUBJ (word "dor")))                             # dor is the word of disjunction,
(be (word (of implication)) (SUBJ (word "ven")))                             # and ven is the word of implication.
(name/VERB (OBJ (kind/NOUN (of word))) too (in grammar) (SUBJ (word "dunon")))    # In grammar, the word dunon names a kind of word too.

(be true (SUBJ negation) (when/SCONJ (be false (SUBJ (child its)))))         # A negation is true when its child is false.
(be true (SUBJ conjunction) (when/SCONJ (be true (SUBJ (child all (of it))))))   # A conjunction is true when all of its children are true.
(be true (SUBJ disjunction) (when/SCONJ (be true (SUBJ (child one/NUM (of it))))))   # A disjunction is true when one of its children is true.
(be false only (SUBJ implication) (when/SCONJ (and (be true (SUBJ (child first/ADJ the))) (be false (SUBJ (child second/ADJ the))))))   # An implication is false only when the first child is true and the second is false.
(be true (SUBJ equivalence) (when/SCONJ (have (OBJ (truth same)) (SUBJ (child two its)))))   # An equivalence is true when its two children have the same truth.
@tree
(if (fall (SUBJ (rain the))) (be wet (SUBJ (ground the))))                  # If the rain falls, the ground is wet.
@prose
(add (OBJ number) (SUBJ (word and)))                                         # s adds numbers,
(link/VERB (OBJ statement) (SUBJ it))                                             # and it joins statements.
(be head (SUBJ it) (in (world both)))                                        # It is a head in both worlds.
(need/VERB (OBJ (symbol (for order))) never (SUBJ talem))                    # Talema never needs symbols for order:
(say (OBJ (child which)) (SUBJ (ending the)))                                # the endings say which children.
@tree
(not (and (fall (SUBJ (rain the))) (be cold (SUBJ it))))                     # Not (the rain falls and it is cold).
@tree
(and (not (fall (SUBJ (rain the)))) (be cold (SUBJ it)))                     # (The rain does not fall) and it is cold.

## (and argument conclusion)
(have (OBJ (and premise conclusion)) (SUBJ argument))                        # An argument has premises and a conclusion.
(be valid (SUBJ argument) (when/SCONJ (follow (from (premise its)) (SUBJ (conclusion its)))))   # An argument is valid when its conclusion follows from its premises.
(be sound/ADJ (SUBJ (argument valid (that (have (OBJ (premise true)))))))    # A valid argument that has true premises is sound.
@verse
(be mortal (SUBJ (man every)))                                               # Every man is mortal.
(be man (SUBJ "Socrates"))                                                   # Socrates is a man.
(be mortal therefore (SUBJ "Socrates"))                                      # Therefore Socrates is mortal.
@prose
(be syllogism (SUBJ (argument this)))                                        # This argument is a syllogism.
(write (OBJ it) before (SUBJ "Aristotle"))                                   # Aristotle wrote it.
(give (OBJ validity) (DAT argument) (SUBJ (form/NOUN its)) only)                  # only its form gives an argument validity.
@verse
(if (fall (SUBJ (rain the))) (be wet (SUBJ (ground the))))                  # If the rain falls, the ground is wet.
(fall (SUBJ (rain the)))                                                     # The rain falls.
(be wet therefore (SUBJ (ground the)))                                       # Therefore the ground is wet.
@prose
(be valid (SUBJ (form this)))                                                # This form is valid.
(be (name (of it)) (SUBJ "modus ponens"))                                    # Its name is modus ponens.

## (and deduction induction abduction)
(infer (OBJ (case/NOUN a)) (from rule) (SUBJ deduction))                          # Deduction infers a case from a rule.
(be certain (SUBJ it))                                                       # It is certain.
(infer (OBJ rule) (from (case/NOUN many)) (SUBJ induction))                       # Induction infers a rule from many cases.
(be certain not (SUBJ it))                                                   # It is not certain.
(infer (OBJ (reason good most the)) (from (thing (that (see (SUBJ we))))) (SUBJ abduction))   # Abduction infers the best reason from the thing that we see.
@verse
(give (OBJ (mark "bove")) (SUBJ deduction))                                  # Deduction gives the mark bove.
(give (OBJ (mark "sere")) (SUBJ induction))                                  # Induction gives the mark sere.
(give (OBJ (mark "pefe")) (SUBJ abduction))                                  # Abduction gives the mark pefe.
@prose
(be white (SUBJ (swan all (that (see (SUBJ they))))))                        # All the swans that they saw were white.
(say (OBJ (be white (SUBJ (swan every)))) (SUBJ they))                       # They said: every swan is white.
(see (OBJ (swan black)) (in "Australia") then (SUBJ they))                  # Then they saw black swans in Australia.
(end/VERB (OBJ (rule their)) (SUBJ (swan one/NUM)))                          # One swan ended their rule.
(prove (OBJ rule) never (SUBJ (case/NOUN many)))                                  # Many cases never prove a rule;
(end/VERB (OBJ it) can (SUBJ (counterexample one/NUM)))                      # one counterexample can end it.

## (and quantifier predicate)
(say (OBJ (many how)) (SUBJ quantifier))                                     # A quantifier says how many.
@verse
(be mortal (SUBJ (man every)))                                               # Every man is mortal.
(be mortal (SUBJ (man some)))                                                # Some man is mortal.
(be mortal (SUBJ (man no)))                                                  # No man is mortal.
@prose
(be (negation (of (statement first/ADJ the))) (SUBJ (statement third the)) not)   # The third statement is not the negation of the first.
(be (be mortal not (SUBJ (man some))) (SUBJ (negation the)))                 # The negation is: some man is not mortal.
(be predicate (SUBJ mortal))                                                 # Mortal is a predicate;
(be individual (SUBJ "Socrates"))                                            # Socrates is an individual.
(give (OBJ property) (DAT individual) (SUBJ predicate))                      # A predicate gives an individual a property.
(have (OBJ domain) (SUBJ (quantifier every)))                                # Every quantifier has a domain:
(be (world (that (count/VERB (in it) (SUBJ it)))) (SUBJ it))                 # the world that it counts in.
@verse
(love (OBJ (person some)) (SUBJ (person every)))                             # Every person loves some person.
@prose
(have (OBJ (meaning two)) (SUBJ (sentence this)))                            # This sentence has two meanings:
(love (OBJ (person own their)) maybe (SUBJ (person every)))                  # maybe every person loves their own person,
(love (OBJ (person same the)) maybe (SUBJ (person every)))                   # maybe every person loves the same person.
(give (OBJ name) (DAT (person the)) (SUBJ you) (for (meaning second/ADJ the)))   # For the second meaning, give the person a name:
@verse
(be person (SUBJ "X"))                                                       # X is a person.
(love (OBJ "X") (SUBJ (person every)))                                       # Every person loves X.
@prose
(open/ADJ (whether (need/VERB (OBJ (rule (for this))) (SUBJ talem))))        # Open: does Talema need a rule for this?

## fallacy
(be (argument (that (and (seem valid) (be valid not)))) (SUBJ fallacy))      # A fallacy is an argument that looks valid and is not valid.
@verse
(if (fall (SUBJ (rain the))) (be wet (SUBJ (ground the))))                  # If the rain falls, the ground is wet.
(be wet (SUBJ (ground the)))                                                 # The ground is wet.
(fall therefore (SUBJ (rain the)))                                           # Therefore the rain falls?
@prose
(be fallacy (SUBJ this))                                                     # This is a fallacy:
(make/VERB (OBJ (ground wet)) (SUBJ (river a)) maybe)                        # maybe a river made the ground wet.
(be (in (premise its)) (SUBJ (conclusion (of (argument circular)))))         # The conclusion of a circular argument is in its premises.
(make/VERB (OBJ statement) true never (SUBJ (authority (of (speaker the))))) # The authority of the speaker never makes a statement true.
(see (OBJ (thing (that (want (OBJ (see (SUBJ it))) (SUBJ it))))) only (SUBJ bias))   # Bias sees only the things that it wants to see.
(search (OBJ (reason (against (conclusion own your)))) (SUBJ you)) # Look for reasons against your own conclusion.

## (and paradox (limit/NOUN (of logic)))
@tree
(be false (SUBJ (sentence this)))                                            # This sentence is false.
@prose
(be false (SUBJ it) (if (be true (SUBJ it))))                                # If it is true, it is false;
(be true (SUBJ it) (if (be false (SUBJ it))))                                # if it is false, it is true.
(be (paradox (of (liar the))) (SUBJ this))                                   # This is the paradox of the liar.
(be (sentence good (of talem)) (SUBJ it))                                    # It is a good sentence of Talema,
(have (OBJ truth) not (SUBJ it))                                             # and it has no truth.
(have (OBJ (child one/NUM)) (SUBJ (tree its)))                               # Its tree has one child:
(point/VERB (at (tree own its)) (SUBJ it))                                        # it points at its own tree.
(prove (OBJ (limit/NOUN (of logic))) before (SUBJ "Gödel"))                  # Gödel proved a limit of logic.
(prove (have (OBJ (statement true (that (prove (OBJ it) (SUBJ it) can not))))) (SUBJ (system consistent every (that (count/VERB)))))   # Proved: every consistent system that counts has true statements that it cannot prove.
(prove (be decidable not (SUBJ (question some))))                            # Proved: some questions are not decidable.
(show (OBJ it) before (SUBJ "Turing"))                                       # Turing showed it.
(be (limit/NOUN (of logic)) (SUBJ (end/NOUN (of logic))) not)                # A limit of logic is not the end of logic.

## (and rule fact)
(build (OBJ (tree (of reason))) (from (and fact rule)) (SUBJ we))            # We build trees of reasons from facts and rules.
@verse
(be (parent (of "Ben")) (SUBJ "Ana"))                                        # Fact: Ana is the parent of Ben.
(be (parent (of "Lut")) (SUBJ "Ben"))                                        # Fact: Ben is the parent of Lut.
(if (and (be (parent (of "Y")) (SUBJ "X")) (be (parent (of "Z")) (SUBJ "Y"))) (be (parent (of (parent (of "Z")))) (SUBJ "X")))   # Rule: if X is the parent of Y and Y is the parent of Z, X is the parent of the parent of Z.
@prose
(ask (OBJ (be (parent (of (parent (of "Lut")))) (SUBJ who))) (SUBJ query))   # A query asks: who is the parent of the parent of Lut?
(be "Ana" (SUBJ (answer the)))                                               # The answer is Ana.
(be (and fact rule) (SUBJ (grammar (of talem))))                     # The grammar of Talema is facts and rules:
(check (OBJ (sentence every (of (book this)))) (SUBJ they))                  # they check every sentence of this book.

## (and symbol meaning)
(have (OBJ meaning) (SUBJ symbol))                                           # A symbol has a meaning.
(be (thing (that (point/VERB (at it) (SUBJ symbol)))) (SUBJ reference))           # A reference is the thing that a symbol points at.
(use (OBJ (word the)) (SUBJ we) (when/SCONJ (speak (about (world the)) (SUBJ we))))   # When we speak about the world, we use words.
(speak (about (word the)) (with "lor") (SUBJ we))                            # We speak about the words with lor.
@verse
(have (OBJ (letter five)) (SUBJ (word "sinov")))                             # The word sinov (snow) has five letters.
(be white (SUBJ snow))                                                       # Snow is white.
(be true (SUBJ (sentence "snow is white")) (if (be white (SUBJ snow))))      # The sentence "snow is white" is true if snow is white.
@prose
(write (OBJ (rule this)) before (SUBJ "Tarski"))                             # Tarski wrote this rule.
(be (world (that (make/VERB (OBJ (statement the)) true (SUBJ it)))) (SUBJ model))   # A model is a world that makes the statements true.

(think (with reason) (in (sentence every)) (SUBJ talem))                      # Talema thinks with reason in every sentence.
