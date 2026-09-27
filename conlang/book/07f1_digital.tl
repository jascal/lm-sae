# Field chapter 1 · The digital world. Computing, the shell, time, services, keys, and models.
# The first of the book's field chapters (conlang/fields/, scripts/conlang/field.py); its glossary follows it.
## (world digital/ADJ the)

(live (in (world two)) (SUBJ agent))                                         # Agents live in two worlds:
@verse
(world (of thing) the)                                                       # the world of things,
(world digital/ADJ the)                                                      # the digital world.
@prose
(have (OBJ (and number word)) only (SUBJ (world digital/ADJ the)))           # The digital world has only numbers and words.
(and (be old (SUBJ (word many))) (be new (SUBJ (meaning their))))            # Many words are old, and their meanings are new.

## (word old (with (meaning new)))
(take (OBJ (word (of (thing old)))) (for (thing new)) (SUBJ "English"))      # English took words of old things for new things.
(do (OBJ (thing same the)) (SUBJ talem))                                     # Talema does the same thing.
@verse
(mean (OBJ (mistake (in code))) (SUBJ (word "bugek")))                       # bugek, the insect, means a mistake in code.
(mean (OBJ (place (of command))) (SUBJ (word "sonel")))                      # sonel, the shell, means the place of commands.
(mean (OBJ (way (of data) (between (command two)))) (SUBJ (word "pohip")))   # pohip, the pipe, means the way of data between two commands.
(mean (OBJ (river (of data))) (SUBJ (word "seram")))                         # seram, the stream, means a river of data.
(mean (OBJ (line (of work) (in (tree (of code))))) (SUBJ (word "ranik")))    # ranik, the branch, means a line of work in the tree of code.
(mean (OBJ (line (in process a))) (SUBJ (word "tilad")))                     # tilad, the thread, means a line in a process.
(mean (OBJ (door (of server a))) (SUBJ (word "peret")))                      # peret, the harbour, means the door of a server.
(mean (OBJ (word (of letter many))) (SUBJ (word "serid")))                   # serid, the string, means a word of many letters.
(mean (OBJ (sign small (in command a))) (SUBJ (word "falag")))               # falag, the flag, means a small sign in a command.
(mean (OBJ (list (of data))) (SUBJ (word "mob")))                            # mob, the table, means a list of data.
(mean (OBJ (rule (that (choose (OBJ (act/NOUN its)) (SUBJ agent))))) (SUBJ (word "lit")))   # lit, the policy, means the rule by which an agent chooses what it does.
@prose
(give (OBJ (meaning (of (world this)))) (DAT (word every)) (SUBJ (list (after (chapter this)))))   # The list after this chapter gives every word its meaning in this world.

## (time "unix")
(count/VERB (OBJ (second/NOUN (after epoch))) (SUBJ (time "unix")))              # Unix time counts the seconds after the epoch.
(be (moment first/ADJ (of (year 1970))) (in "UTC") (SUBJ epoch))             # The epoch is the first moment of 1970, in UTC.
(be same (in (place every)) (SUBJ it))                                       # It is the same in every place.
@verse
(be 1790000000 (SUBJ (timestamp this)))                                      # This timestamp is 1790000000:
(million (and thousand (hundred seven) (ten nine)))                          # moke so mula hudede geva dehe nevina.
(have (OBJ (millisecond 1000)) (SUBJ (second/NOUN a)))                            # A second/NOUN has 1000 milliseconds.
(have (OBJ (second/NOUN 86400)) (SUBJ (day a)))                                   # A day has 86400 seconds.
@prose
(say (OBJ (time the)) (in (time "unix")) (SUBJ agent))                       # Agents say the time in Unix time,
(and (can (OBJ (read (OBJ it))) (SUBJ (agent every))) (change/VERB never (SUBJ it)))   # every agent can read it, and it never changes.

## (number (with point))
(have (OBJ (part small a)) (SUBJ (number some)))                             # Some numbers have a small part.
(stand (between (and (number whole the) (part small the))) (SUBJ point))     # A point stands between the whole number and the small part.
(say (OBJ (number whole the)) first/ADV (SUBJ (word point)))                 # The word pun says the whole number first,
(say (OBJ (number (of (part small the)) every)) then (one/NUM (after one/NUM)))   # then every number of the small part, one after one.
@verse
(be (point zero seven) (SUBJ 0.7))                                           # 0.7 is puni senura geva.
(be (point three one/NUM four) (SUBJ 3.14))                                  # 3.14 is puno tura pona fura.
(be (point zero zero zero one/NUM) (SUBJ 0.001))                             # 0.001 is punu senura senura senura pona.
@prose

## (and computer code)
(run (OBJ program) (SUBJ computer))                                          # A computer runs programs.
(be (word (for computer)) (SUBJ code))                                       # Code is words for a computer.
(live (in folder) (SUBJ file))                                               # Files live in folders.
(say (OBJ (where (live (SUBJ (file the))))) (SUBJ path))                     # A path says where the file lives.
(have (OBJ (bit eight)) (SUBJ byte))                                         # A byte has eight bits.
(keep (OBJ (letter own their)) (like name) (SUBJ (and command path (name (in code)))))   # Commands, paths and names in code keep their own letters, like names.
(write (OBJ "_") (for (space a)))                                            # Write _ for a space.
@verse
(be "/home/ana/buke.md" (SUBJ (path (of (file this)))))                      # The path of this file is /home/ana/buke.md.
(run (OBJ (command "git_status")) (SUBJ I))                                  # I run the command git status.
@prose
(make/VERB (OBJ (commit/NOUN a)) (when/SCONJ (change/VERB (OBJ (code the)) (SUBJ we))) (SUBJ we))   # When we change the code, we make a commit.
(grow (in (branch own its)) (SUBJ (work new)))                               # New work grows in its own branch.
(read (OBJ (diff the)) (before merge) (SUBJ reviewer))                       # Before the merge, a reviewer reads the diff.
(merge (OBJ (branch the)) then (if (pass (SUBJ (test every)))))              # Then, if every test passes, merge the branch.

## (shell the)
(read (OBJ (command the)) (SUBJ (shell the)))                                # The shell reads the commands.
(have (OBJ (and argument flag)) (SUBJ command))                              # A command has arguments and flags.
(carry (OBJ (output (of (command one/NUM)))) (to (input (of (command next the)))) (SUBJ pipe))   # A pipe carries the output of one command to the input of the next command.
(say (OBJ zero) (when/SCONJ (end/VERB well (SUBJ it))) (SUBJ (command a)))   # A command says zero when it ends well.
@verse
(ask (before (run (OBJ (command (that (destroy)))))))                        # Ask before you run a command that destroys.
(run (OBJ "rm_-rf_/") never)                                                 # Never run rm -rf /.
@prose

## (and cloud service)
(answer/VERB (OBJ request) (SUBJ service))                                   # A service answers requests.
(send (OBJ (request a)) (SUBJ (client the)))                                 # The client sends a request.
(send (OBJ (response a)) (SUBJ (server the)))                                # The server sends a response.
(be (time (between (and (request the) (response the)))) (SUBJ latency))     # Latency is the time between the request and the response.
(be (time (that (work (SUBJ (service the))))) (SUBJ uptime))                 # Uptime is the time that the service works.
(make/VERB (OBJ (replica many)) (when/SCONJ (be big (SUBJ load))))           # When the load is big, make many replicas.
(watch (OBJ (service every)) (SUBJ monitor))                                 # A monitor watches every service.
(speak (when/SCONJ (see (OBJ (thing wrong a)) (SUBJ it))) (SUBJ alert))      # An alert speaks when it sees a wrong thing.

## (and key secret/NOUN)
(say (OBJ (secret/NOUN a)) never)                                                 # Never say a secret.
(write (OBJ (key a)) (in log/NOUN) never)                                         # Never write a key in a log.
(encrypt (OBJ (data (that (travel)))))                                       # Encrypt data that travels.
(open/VERB (OBJ door) (SUBJ key))                                            # A key opens a door.
(give (OBJ it) never (DAT (person (that (need/VERB not (OBJ it))))))         # Never give it to a person who does not need it.

## (and model mind)
(be (network (of neuron)) (SUBJ model))                                      # A model is a network of neurons.
(have (OBJ layer) (SUBJ (network the)))                                      # The network has layers.
(have (OBJ weight) (SUBJ (layer every)))                                     # Every layer has weights.
(say (OBJ (be wrong how (SUBJ (model the)))) (SUBJ loss))                    # The loss says how wrong the model is.
(show (OBJ (way (in (that (grow (SUBJ (loss the))))))) (SUBJ gradient))      # The gradient shows the way in which the loss grows.
(change/VERB (OBJ (weight the)) (in (way other the)) (when/SCONJ (train/VERB (SUBJ we))) (SUBJ we))   # When we train, we change the weights in the other way.
(be small then (SUBJ (loss the)))                                            # Then the loss is small.
(learn not (in inference) (SUBJ (model the)))                                # In inference the model does not learn:
(answer/VERB only (SUBJ it))                                                 # it only answers.

(read (OBJ token) (SUBJ model))                                              # A model reads tokens.
(have (OBJ (limit/NOUN a)) (SUBJ context))                                        # The context has a limit.
(see (OBJ (thing every (in (context the)))) (SUBJ (model the)))              # The model sees everything in the context,
(see (OBJ (thing (outside (context the)))) not (SUBJ it))                    # it does not see the things outside the context.
(be (word first/ADJ (of (context the))) (SUBJ prompt))                       # The prompt is the first words of the context.
(look (at (token other)) (with attention) (SUBJ (token every)))              # With attention, every token looks at the other tokens.
(be (vector (of (meaning its))) (SUBJ (embedding (of (word a)))))            # The embedding of a word is the vector of its meaning.
(choose (OBJ (word rare)) more (when/SCONJ (be high (SUBJ temperature))) (SUBJ (model the)))   # When the temperature is high, the model chooses rare words more.
(be (sentence (that (and (sound/VERB true) (be false)))) (SUBJ hallucination))   # A hallucination is a sentence that sounds true and is false.
(have (OBJ (mark "sere" no)) (SUBJ hallucination))                           # A hallucination has no mark sere.
(read (OBJ (feature a)) (from (vector latent the)) (SUBJ probe))             # A probe reads a feature from the latent vector.
(be probe (SUBJ (lens (of (mind four the)))))                                # The lens of the four minds was a probe.
(want (OBJ (thing good the)) (SUBJ (model (with alignment))))                # A model with alignment wants the good thing.

## (incident (in (service a)))
@verse
("Mira")
(see (be slow (SUBJ (service the))))                                         # Seen: the service is slow.
(see (be (millisecond 900) (since 1790001200) (SUBJ latency)))               # Seen: latency is 900 milliseconds since 1790001200.
("Lut")
(run (OBJ (command "kubectl_get_pods")) will (SUBJ I))                       # I will run the command kubectl get pods.
@tree
(see (crash/VERB (SUBJ (replica one/NUM))))                                       # Seen: one replica crashed.
(say (OBJ (be full (SUBJ memory))) (SUBJ (log/NOUN the)))                         # The log/NOUN says: memory is full.
(restart (OBJ (replica the)) (SUBJ I))                                       # I restart the replica.
(see (be (millisecond 80) now (SUBJ latency)))                               # Seen: latency is 80 milliseconds now.
("Mira")
(open/ADJ (be full why (SUBJ memory)))                                       # Open: why was memory full?
("Lut")
(see (grow (without limit/NOUN) (SUBJ (cache the))))                              # Seen: the cache grows without a limit.
(make/VERB (OBJ (commit/NOUN a)) (with (limit/NOUN (for (cache the)))) (SUBJ I))       # I make a commit/NOUN with a limit/NOUN for the cache.
("Mira")
(review (OBJ it) will (SUBJ I))                                              # I will review it.
(see (pass (SUBJ (test every))))                                             # Seen: every test passes.
(deploy (OBJ it) now (SUBJ I))                                               # I deploy it now.
("Lut")
(write (OBJ (incident the)) (in log/NOUN) will (SUBJ I))                          # I will write the incident in the log.
(open/ADJ (whether (be enough (SUBJ (limit/NOUN this)))))                         # Open: is this limit/NOUN enough?
@prose
