# Field chapter 2 · Mathematics. Talema's grammar is already mathematical notation: the head first, its children
# counted (Łukasiewicz's Polish notation). Operators are relator heads; functions are literal heads with arguments.
## mathematics

(speak (OBJ mathematics) already (SUBJ talem))                               # Talema already speaks mathematics.
(speak first/ADV (SUBJ (head the)))                                          # The head speaks first,
(count/VERB (OBJ (child its)) (SUBJ (ending its)))                           # its ending counts its children.
(write (OBJ (number the)) (with (head first/ADJ)) before (SUBJ "Łukasiewicz"))   # Łukasiewicz wrote numbers with the head first.
@tree
(be four (SUBJ (and two two)))                                               # Two and two are four.

## (act/NOUN (of number))
(add (OBJ (child its)) (SUBJ (word and)))                                    # s adds its children.
(take (OBJ (child second/ADJ the)) (from (child first/ADJ the)) (SUBJ (word minus)))   # menos takes the second child from the first.
(be (under zero) (with (child one/NUM)) (SUBJ it))                           # With one child it is below zero.
(multiply (OBJ (child its)) (SUBJ (word times)))                             # mapor multiplies its children.
(divide (OBJ (child first/ADJ the)) (by (child second/ADJ the)) (SUBJ (word divided)))   # dided divides the first child by the second.
(be (root square the) (with (child one/NUM)) (SUBJ root))                    # raris with one child is the square root;
(say (OBJ (root which)) (with (child two)) (SUBJ (child first/ADJ the)))     # with two children, the first child says which root.
(say (OBJ equal) (SUBJ (word be)))                                           # b says equal.
@verse
(be four (SUBJ (minus seven three)))                                         # 7 − 3 = 4
(be (and ten two) (SUBJ (times three four)))                                 # 3 × 4 = 12
(be three (SUBJ (divided (and ten two) four)))                               # 12 / 4 = 3
(be (and thousand (ten two) four) (SUBJ (power two ten)))                    # 2^10 = 1024
(be three (SUBJ (root nine)))                                                # √9 = 3
(be two (SUBJ (root three eight)))                                           # ∛8 = 2
(be (minus five) (SUBJ (minus three eight)))                                 # 3 − 8 = −5
(be (divided one/NUM two) (SUBJ half))                                       # a half is 1/2
@prose
(need/VERB (OBJ (symbol (for order))) never (SUBJ talem))                    # Talema never needs symbols for order:
(say (OBJ (order the)) (SUBJ (ending the)))                                  # the endings say the order.
@tree
(times (and one/NUM two) three)                                              # (1 + 2) × 3
@tree
(and one/NUM (times two three))                                              # 1 + 2 × 3

## (and variable/NOUN function)
(keep (OBJ (letter own their)) (like name) (SUBJ (and variable/NOUN function)))   # Variables and functions keep their own letters, like names.
(be head (SUBJ function))                                                    # A function is a head;
(be (argument its) (SUBJ (child its)))                                       # its children are its arguments.
@verse
(be (power "x" two) (SUBJ ("f" "x")))                                        # f(x) = x²
(be (and "x" "y") (SUBJ ("g" "x" "y")))                                      # g(x, y) = x + y
@prose
(be five (SUBJ (and "x" two)))                                               # x + 2 = 5:
(be (unknown/NOUN the) (SUBJ "x"))                                                # x is the unknown.
(solve (OBJ (equation the)))                                                 # Solve the equation:
(be three (SUBJ "x"))                                                        # x = 3.

## (and set/NOUN element)
(have (OBJ element) (SUBJ set/NOUN))                                              # A set/NOUN has elements.
(have (OBJ (element no)) (SUBJ (set/NOUN empty the)))                             # The empty set/NOUN has no elements.
(have (OBJ (element every (of (set/NOUN two the)))) (SUBJ union))                 # A union has every element of the two sets.
(have (OBJ (element (that (be (in (set/NOUN both)))))) only (SUBJ intersection))  # An intersection has only the elements that are in both sets.
(be (in (set/NOUN big the)) (SUBJ (element every (of subset))))                   # Every element of a subset is in the big set.

## geometry
(have (OBJ (and line angle)) (SUBJ triangle))                                # A triangle has lines and angles.
(be (degree 180) (SUBJ (sum (of (angle (of triangle))))))                    # The sum of the angles of a triangle is 180 degrees.
(meet never (SUBJ (line parallel two)))                                      # Two parallel lines never meet.
@verse
(be (times "pi" (power "r" two)) (SUBJ (area (of circle))))                  # The area of a circle is π r².
(be (and (power "a" two) (power "b" two)) (SUBJ (power "c" two)))            # a² + b² = c²
@prose

## (and proof theorem)
(stand first/ADV (SUBJ axiom))                                               # Axioms stand first;
(prove not (SUBJ we) (OBJ they))                                             # we do not prove them.
(give (OBJ name) (DAT (thing a)) (SUBJ definition))                          # A definition gives a thing a name.
(be (thing (that (prove (from axiom) (SUBJ we)))) (SUBJ theorem))            # A theorem is a thing that we prove from axioms.
(be (tree (of reason)) (SUBJ proof))                                         # A proof is a tree of reasons.
(be (mark (of theorem)) (SUBJ (word "bove")))                                # bove is the mark of a theorem.
(be (mark (of conjecture)) (SUBJ (word "pefe")))                             # pefe is the mark of a conjecture.

## (end/VERB never (SUBJ (prime/NOUN the)))
(be prime/NOUN (SUBJ (number (that (divide (SUBJ (and one/NUM it)) only)))))      # A prime/NOUN is a number that only one and itself divide.
(prove (end/VERB never (SUBJ (prime/NOUN the))))                                 # Proved: the primes never end.
(show (OBJ it) before (SUBJ "Euclid"))                                       # Euclid showed it.
@tree
(assume (OBJ (end/VERB (SUBJ (prime/NOUN the)))))                                # Assume: the primes end.
(make/VERB (OBJ (product (of (prime/NOUN all the)))))                             # Make the product of all the primes,
(add (OBJ one/NUM) (to it))                                                  # and add one to it.
(be "N" (SUBJ (number this)))                                                # This number is N.
(be one/NUM (SUBJ (remainder (of (divided "N" (prime/NOUN every))))))             # N divided by every prime/NOUN leaves remainder one:
(divide (OBJ "N") (SUBJ (prime/NOUN no)))                                     # no prime/NOUN divides N.
(but (have (OBJ (factor prime/NOUN a)) (SUBJ (number every (greater (than one/NUM))))))   # But every number greater than one has a prime/NOUN factor.
(be (in (list the)) not (SUBJ (factor this)))                                # This factor is not in the list.
(be contradiction (SUBJ this))                                               # This is a contradiction.
(be false therefore (SUBJ (thing (that (assume (SUBJ we))))))                # Therefore the thing we assumed is false,
(end/VERB never (SUBJ (prime/NOUN the)))                                         # and the primes never end.

(mean (OBJ (without end/NOUN)) (SUBJ (word "finit")))                       # The word finit means without end: infinite.
(mean (OBJ (with end/NOUN)) (SUBJ (word "nilik")))                           # The word nilik means with an end: finite.
(read (OBJ they) slowly (if (read (OBJ "Español") (SUBJ you))))              # If you read Spanish, read them slowly.

## (conjecture (of "Goldbach"))
(open/ADJ (be (sum (of (prime/NOUN two))) (SUBJ (number even/ADJ every (greater (than two))))))   # Open: every even/ADJ number greater than two is the sum of two primes.
(see (be true (for (number even/ADJ every (under (times four (power ten (and ten eight))))))))   # Seen: it is true for every even/ADJ number below 4 × 10^18.
(prove (OBJ it) yet (SUBJ nobody))                                       # Nobody has proved it yet.
(stay (with (mark "pefe")) (until (proof the)) (SUBJ it))                    # It stays with the mark pefe until the proof.

## calculus
(say (OBJ (fast/ADJ how (change/VERB (SUBJ (function the))))) (SUBJ derivative))   # The derivative says how fast the function changes.
(be (sum (of (part small many))) (SUBJ integral/NOUN))                            # An integral/NOUN is the sum of many small parts.
(approach/VERB (OBJ (limit/NOUN its)) (SUBJ sequence))                            # A sequence approaches its limit.
(reach (OBJ it) never maybe (SUBJ it))                                       # Maybe it never reaches it.

## (and chance probability)
(be (between (and zero one/NUM)) (SUBJ probability))                         # A probability is between zero and one.
(be (divided one/NUM two) (SUBJ (probability (of (side one/NUM (of coin))))))   # The probability of one side of a coin is 1/2.
(be (divided (sum the) (number (of element))) (SUBJ average/NOUN))                # The average/NOUN is the sum divided by the number of elements.

(count/VERB (in (word every)) (SUBJ talem))                                  # Talema counts in every word.
