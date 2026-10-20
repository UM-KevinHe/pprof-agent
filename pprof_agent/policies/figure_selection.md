# Choosing the figures

## What you are doing
You choose which displays from a fixed catalogue go into the report for this analysis, and say why. The harness draws them; you only name them.

## What the system is
Each display in the catalogue states the question it answers, what result it needs, when it is informative, and when it is refused. A display is drawn of one result, named by its id: a test under the theoretical null (`test:<name>`), a test under the empirical null (`en:<name>`), the fitted model (`fit`), the model evaluation (`evaluation`). The harness refuses a display whose requirements the result does not meet and records why. Methods and measures call for different displays: choose what this analysis needs, not every display.

## Before you decide
In `reasoning`, go through the catalogue and say for each display whether it answers a question this analysis raises: the reported test and reference, the providers that change status under alternatives, the degeneracy of the small providers, the fit of the risk adjustment. Name the result each chosen display is drawn of.

## The fields
  figures   up to six items, each with `display` (a catalogue name), `of` (a result id), and `reason` (one sentence, in words, without numbers).

## What you must not do
Do not name a display that is not in the catalogue. Do not draw a display of a result that does not exist. Do not choose a display whose card says it is refused for this kind of test. Do not write numbers in the reasons.
