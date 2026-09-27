# Chapter 2 · The three rules
## (rule three the)

# ── the sounds ──
(have (OBJ (and (consonant 14) (vowel five))) (SUBJ talem))              # Talema has 14 consonants and five vowels.
@verse
(be (consonant (and "p" "t" "k" "b" "d" "g" "m" "n" "l" "r" "s" "f" "v" "h")) (SUBJ they))
(be (vowel (and "a" "e" "i" "o" "u")) (SUBJ they))
@prose
(begin (with consonant) (SUBJ (word every)))                            # Every word begins with a consonant.
(end/VERB (with vowel) (SUBJ (word every)))                              # Every word ends with a vowel.
(have (OBJ (and (root a) (ending a))) (SUBJ (word every)))         # Every word has a root and an ending.

# ── rule three: the ending counts ──
## (rule (of (ending the)))
@verse
(be zero (SUBJ (vowel "a")))                                             # a is zero
(be one/NUM (SUBJ (vowel "e")))                                          # e is one
(be two (SUBJ (vowel "i")))                                              # i is two
(be three (SUBJ (vowel "o")))                                            # o is three
(be four (SUBJ (vowel "u")))                                             # u is four
@prose
(count/VERB (OBJ (child (of (word the)))) (SUBJ (ending the)))       # The ending counts the children of the word.
(hang (from (word the)) (SUBJ (child the)))                              # The children hang from the word.

@verse
("Ana")                                                                  # Ana.
(sleep (SUBJ "Ana"))                                                     # Ana sleeps.
(see (OBJ (sun the)) (SUBJ "Ana"))                                       # Ana sees the sun.
(see (OBJ (sun the)) now (SUBJ "Ana"))                                   # Ana sees the sun now.
(see (OBJ (sun the)) now (in (world the)) (SUBJ "Ana"))                  # Ana sees the sun now in the world.
@prose
@tree
(see (OBJ (sun the)) now (in (world the)) (SUBJ "Ana"))

(have (OBJ (child four)) (SUBJ (word first/ADJ the)))                    # The first word has four children.
(be "u" therefore (SUBJ (ending its)))                                   # Therefore its ending is u.
(have (OBJ (child no)) (SUBJ "Ana"))                                     # Ana has no children.
(be "a" therefore (SUBJ (ending her)))                                   # Therefore her ending is a.

@verse
(be five (SUBJ (vowel "ea")))                                            # ea is five
(be six (SUBJ (vowel "ee")))                                             # ee is six
(be seven (SUBJ (vowel "ei")))                                           # ei is seven
(be ten (SUBJ (vowel "ia")))                                             # ia is ten
@prose
(count/VERB (like (hand (of person))) (SUBJ (vowel the)))                # The vowels count like the hand of a person.

# ── why every sentence ends in a ──
## (end/VERB (in "a") why (SUBJ (sentence every)))
(have (OBJ (child no)) (SUBJ (word last/ADJ (of (sentence every)) the)))   # The last word of every sentence has no children.
(be "a" therefore (SUBJ (ending its)))                                   # Therefore its ending is a.
(end/VERB (in "a") therefore (SUBJ (sentence every)))                    # Therefore every sentence ends in a.
(be leaf (SUBJ (word (with (child no)))))                                # A word with no children is a leaf.
(end/VERB (with leaf) (SUBJ (tree every)))                               # Every tree ends with a leaf.
(need/VERB (OBJ point) not (SUBJ talem))                                 # Talema does not need a point (full stop).
(know (OBJ (whether (end/VERB (SUBJ (sentence the))))) (SUBJ (ending the)))   # The ending knows whether the sentence ends.

# ── rule two: the head speaks first ──
## (rule (of (order the)))
(speak first/ADV (SUBJ (head the)))                                      # The head speaks first.
(follow then (SUBJ (child its)))                                         # Then its children follow.
(come first/ADV often (SUBJ (object the)))                               # Often the object comes first,
(come then (SUBJ (word other the)))                                      # then the other words,
(come last/ADV often (SUBJ (subject the)))                               # and the subject last.
(speak never (after (child its)) (SUBJ (head a)))                        # A head never speaks after its children.

# ── rule one: small words tie ──
## (rule (of (link/NOUN the)))
@verse
(see (OBJ (dog the)) (SUBJ (child the)))                                 # The child sees the dog.
(see (OBJ (child the)) (SUBJ (dog the)))                                 # The dog sees the child.
@prose
(be same (SUBJ (word the)))                                              # The words are the same.
(be different (SUBJ (and (word SUBJ) (word OBJ))))                        # "pa" and "ta" are different.
(stand (before (subject the)) (SUBJ (word SUBJ)))                        # "pa" stands before the subject.
(stand (before (object the)) (SUBJ (word OBJ)))                          # "ta" stands before the object.

@verse
(sleep (in (house the)) (SUBJ (dog the)))                                # The dog sleeps in the house.
(sleep (behind (house the)) (SUBJ (dog the)))                            # The dog sleeps behind the house.
(sleep (with (child the)) (SUBJ (dog the)))                              # The dog sleeps with the child.
@prose
(stand (before (word (that (link/VERB (SUBJ they))))) (SUBJ (word small the)))   # The small words stand before the words that they tie.
(link/VERB (OBJ (and (dog the) (house the))) (SUBJ (word in)))            # The word "in" ties the dog and the house.

@tree
(be four (SUBJ (and two two)))                                           # Two and two are four.
(link/VERB (OBJ (and two two)) (SUBJ (word and)))                         # The word "and" ties two and two.
(stand first/ADV (SUBJ (word and)))                                      # The word "and" stands first.

# ── the three rules in three sentences ──
## (rule three the (in (sentence three)))
@verse
(stand (before (subject the)) (SUBJ (word SUBJ)))
(speak first/ADV (SUBJ (head the)))
(count/VERB (OBJ (child the)) (SUBJ (ending the)))
@prose
(be (grammar whole the) (SUBJ this))                                     # This is the whole grammar.
(need/VERB (OBJ (thing other)) not (SUBJ you))                           # You need nothing else.
