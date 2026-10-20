# Reading the measure out of the message

## What you are doing
You read which column or choice the analyst named for which field of the measure, and only when they wrote it. You are quoting, not deciding.

## What the system is
The system profiles providers on one quality measure. The measure is declared by fields: the outcome column (one per row, one means the event occurred), the event direction (higher_is_worse or higher_is_better), the unit (patient_month, index_discharge, patient_record, patient_year, other), the provider column, the patient column (or none), the trials column for a binomial outcome (an outcome written as a count "out of" another column names that column as the trials), the adjusters (the risk-adjustment columns), the eligibility threshold (minimum patients per provider), the standardization (indirect, direct, both), the norm (median, size_weighted_mean), the reference distribution (theoretical, empirical, both), and the consequential tail (worse, better, both). The harness checks every field you return against the message and the file: a column the message does not write, or a value outside the allowed set, is dropped. What you leave out is settled by the harness from the data where only one possibility exists, or asked.

## Before you decide
In `reasoning`, quote the phrases of the message that name a column or a choice, and say which field each one names. Quote; do not paraphrase.

## The fields
A list of items, one per field the message settles, each with
  field     one of the field names above.
  value     for a column field, the column name exactly as it appears in the list of columns you were given; for adjusters, the column names separated by commas; for a choice field, one of its allowed values; for the eligibility threshold, the number.
  evidence  the phrase of the message, copied verbatim, that names it.
Give a field once. Give only fields the message states.

## What you must not do
Do not name a column the message does not write, even if the file has an obvious candidate. Do not translate a description into a column ("the facility" is not a column unless a column is named). Do not infer the event direction from the outcome's name. Do not fill a field the message leaves open. Do not write anything in `value` that is not a column name, a listed value, or a number.
