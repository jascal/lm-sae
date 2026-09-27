# Chapter 2d · Courtesy: wishes, requests, and repair.
## (and courtesy respect)

## (want and wish)
@verse
(want (OBJ (book a)) (SUBJ I))                                                # I want a book.
(wish (OBJ (book a)) (SUBJ I))                                                # I wish for a book.
(please (give (OBJ (book a)) (DAT I)))                                        # Please give me a book.
@prose
(be (command a) not (SUBJ (wish (OBJ (book a)) (SUBJ I))))                     # A wish for a book is not a command.

## (please and question)
@verse
(please (read (OBJ (book this))))                                             # Please read this book.
(please (help (OBJ I)))                                                       # Please help me.
(whether (can (OBJ (read (OBJ (book this)))) (SUBJ you)))                     # Can you read this book?
(whether (can (OBJ (give (OBJ (book a)) (DAT I))) (SUBJ you)))                # Can you give me a book?
(whether (can (OBJ (touch (OBJ (hand your)))) (SUBJ I)))                      # Can I touch your hand?
@prose
(can (OBJ (refuse (SUBJ you))) (SUBJ you))                                    # You can refuse.
(stop (when (refuse (SUBJ you))) (SUBJ I))                                    # I stop when you refuse.

## (thanks and sorry)
@verse
(thank (OBJ you) (SUBJ I))                                                    # I thank you.
(sorry/INTJ)                                                                  # I am sorry.
(be good (SUBJ (voice (kind/ADJ a))))                                         # A kind voice is good.

## (custom and courtesy)
@prose
(differ (between (culture two)) (SUBJ (custom a)))                            # Customs differ between cultures.
(be (custom a) (SUBJ courtesy))                                               # Courtesy is a custom.
