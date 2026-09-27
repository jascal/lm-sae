# Chapter 2c · How to say more. Conventions the reviews asked for: time and number, questions, focus,
# commands and address, and the senses of a few core words. None is a new rule; each is a use of the three.
## (say (OBJ more) how)

# ── time and number: unsaid unless a word says them ──
## (and time number)
@verse
(sleep (SUBJ (dog the)))                                                     # The dog sleeps / slept / will sleep.
(sleep before (SUBJ (dog the)))                                              # The dog slept.
(sleep now (SUBJ (dog the)))                                                 # The dog sleeps now.
(sleep will (SUBJ (dog the)))                                                # The dog will sleep.
@prose
(say (OBJ (time no)) (SUBJ (sentence (without (word (of time))))))           # A sentence without a word of time says no time.
@verse
(see (OBJ dog) (SUBJ I))                                                     # I see a dog / dogs.
(see (OBJ (dog one/NUM)) (SUBJ I))                                           # I see one dog.
(see (OBJ (dog two)) (SUBJ I))                                               # I see two dogs.
(see (OBJ (dog many)) (SUBJ I))                                              # I see many dogs.
@prose
(say (OBJ (number no)) (SUBJ (word (without (word (of number))))))           # A word without a word of number says no number.
(be good (SUBJ this))                                                        # This is good:
(say (OBJ (thing (that (know (SUBJ you))))) only (SUBJ you))                 # you say only the thing that you know.

# ── questions: three shapes, three jobs ──
## (question the)
@verse
(whether (sleep (SUBJ (dog the))))                                           # Does the dog sleep?
@prose
(ask (SUBJ it))                                                              # It asks.
(stand alone (SUBJ (word "tob")))                                          # The word tob stands alone:
(be question (SUBJ (sentence the)))                                          # the sentence is a question.
@verse
(know (OBJ (whether (sleep (SUBJ (dog the))))) not (SUBJ I))                 # I do not know whether the dog sleeps.
@prose
(ask not (SUBJ it))                                                          # It does not ask.
(tell (OBJ (thing (that (know not (SUBJ I))))) (SUBJ it))                    # It tells the thing that I do not know.
@verse
(open/ADJ (whether (sleep (SUBJ (dog the)))))                                # Open: does the dog sleep?
@prose
(say (OBJ (know (OBJ it) not yet (SUBJ we))) (SUBJ it))                      # It says: we do not know it yet.
(be (mark a) (SUBJ it))                                                      # It is a mark.

@verse
(see (OBJ what) (SUBJ you))                                                  # What do you see?
(see (OBJ (dog the)) (SUBJ I))                                               # I see the dog.
(sleep (where) (SUBJ (dog the)))                                             # Where does the dog sleep?
(sleep (in (house the)) (SUBJ (dog the)))                                    # The dog sleeps in the house.
@prose
(stand (where (stand (SUBJ (answer the)))) (SUBJ (word (of question))))      # The word of the question stands where the answer stands.

# ── focus: the child nearest the head carries the weight ──
## (weight/NOUN (of (order the)))
@verse
(see (OBJ (dog the)) (SUBJ (child the)))                                     # The child sees the dog.
(see (SUBJ (child the)) (OBJ (dog the)))                                     # It is the child who sees the dog.
@prose
(mean (OBJ (thing same the)) (SUBJ (sentence two the)))                      # The two sentences mean the same thing.
(but (carry (OBJ (weight/NOUN the)) (SUBJ (child (near (head the))))))       # But the child near the head carries the weight.
@verse
(see (OBJ (dog the)) (SUBJ who))                                             # Who sees the dog?
(see (SUBJ (child the)) (OBJ (dog the)))                                     # The CHILD sees the dog.
(sleep (in (house the)) (SUBJ (dog the)))                                    # The dog sleeps in the house.
(sleep (SUBJ (dog the)) (in (house the)))                                    # The DOG sleeps in the house.
@prose
(stand (near (head the)) often (SUBJ (answer the)))                          # The answer often stands near the head.

# ── commands and address ──
## (and command call/NOUN)
(stand (under please) (SUBJ (command a)))                                    # A command stands under les.
(stand (under to) (SUBJ (person (that (hear)))))                             # The person who hears stands under ge.
@verse
(please (sleep))                                                             # Please sleep. / Sleep!
(please (sleep) (to (child the)))                                            # Child, sleep!
(please (read (OBJ (book this))) (to (reader the)))                          # Reader, read this book!
@prose
(can (OBJ (be (command a))) (SUBJ (sentence (without subject))))             # A sentence without a subject can be a command.
(make/VERB (OBJ it) sure (SUBJ (word "les")))                               # The word les makes it sure.

# ── words that are easy to confuse ──
## (word (that (look/VERB same)))
(be (tree (of word) a) (SUBJ sentence))                                      # A sentence (feras) is a tree of words.
(be (symbol (for (sound a))) (SUBJ letter))                                  # A letter (let) is a symbol for a sound.
(be (word (under SUBJ)) (SUBJ subject))                                      # The subject (tum) is the word under pe.
(be (word (under OBJ)) (SUBJ object))                                        # The object (behet) is the word under te.
(be numeral (SUBJ zero))                                                     # senur is a numeral: zero dogs.
(be noun (SUBJ =nuler))                                                      # nuler is a noun: a zero, the sign.
(be preposition (SUBJ to))                                                   # g is a preposition: to the house.
(stand (before verb) (SUBJ to/PART))                                         # pat stands before a verb: to go.
(need/VERB (OBJ it) rarely (SUBJ talem))                                     # Talema rarely needs it.
(be (word same the) (SUBJ (and =kas =pas)))                                  # kas and pas are the same word:
(write (OBJ (word "daß")) (in (time old the)) (SUBJ "Deutsch"))              # German wrote daß in the old time.
(use/VERB (OBJ =kas))                                                        # Use kas.
(mean (OBJ answer/VERB) (SUBJ (word "never")))                               # The word never means "answer".
(mean (OBJ never) (SUBJ (word "ner")))                                       # The word ner means "never".
(read (OBJ they) slowly (if (read (OBJ "English") (SUBJ you))))              # If you read English, read them slowly.
