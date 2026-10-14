# Conversation summary and attribution

## Starting question

The user asked how traditional ML, deep learning, pretrained models, and modern
AI systems relate, and whether pretrained models are replacing classical ML.
We clarified that the business prediction task can remain the same while the
learning method changes, and that accuracy, training cost, inference cost, and
operational requirements must be measured separately.

## Tabular prediction

A table contains examples in rows and features in columns. Supervised prediction
uses known outcomes for past rows to predict outcomes for new rows. Predicting
a salary or price is regression; predicting a category is classification.

Traditional models can be trained once, saved, and reused. Pretrained table
models such as TabPFN can instead use labeled examples as context. A language
model can also consume serialized rows, but that is a different representation
and training objective from a specialized tabular model.

## Gemma adaptation

The user asked how to adapt an open-weight Gemma E4B model. We discussed
serializing a row to text and using supervised adaptation, potentially through
LoRA. We also discussed a hybrid where text representations feed a conventional
predictor together with numerical features.

## The user's digit proposal

The user proposed encoding a number such as 180000 as `+1.800e+05` and selecting
eight variable slots:

1. Mantissa sign.
2. First mantissa digit.
3. Second mantissa digit.
4. Third mantissa digit.
5. Fourth mantissa digit.
6. Exponent sign.
7. First exponent digit.
8. Second exponent digit.

The punctuation `.` and `e` is deterministic. The original idea selects slots
sequentially, analogous to choosing among multiple-choice token options using
logits and softmax. This can reuse a language model's vocabulary head with
appropriate constrained decoding. Actual tokenization must be checked; a
character is not necessarily a single model token.

The user then asked whether these choices could be made in one forward pass.
The assistant suggested reading one shared Gemma hidden representation with
small parallel categorical heads. Parallel slot outputs do not condition on
the selected values of earlier slots in the same way as sequential decoding.

## Exact attribution of the regression-head suggestion

The user asked: "we could do this one forward pass make it efficient tabular model".

In the response to that message, the assistant first suggested comparing the
digit-head design against "a single numerical regression head on the same Gemma
backbone." This suggestion was the assistant's. It was introduced as an
experimental control, not as part of the user's original digit design.

We later clarified:

- A language head projects hidden representations to vocabulary logits.
- The proposed small classification heads project hidden representations to
  scores for signs and digits.
- A direct regression head projects a hidden representation to a numerical value.
- LoRA is an adaptation method; it is separate from the choice of output head.

The comparison tests whether improvements come from Gemma's representations or
from the user's numerical output encoding.

## Experiment authorization

The user requested a concrete production-oriented comparison, then authorized:

- A new local repository, to be pushed later.
- Documentation of what, why, how, and this conversation before training.
- Periodic findings suitable for a later engineering blog.
- Modal compute using supplied credentials with **a $30 total spending limit**.

Credentials are intentionally omitted from this summary and all repository files.

## What is still a hypothesis

We have not established that this Gemma design beats tree models or a direct
regression head. One backbone pass avoids an autoregressive output loop but
still requires running the Gemma backbone. Four mantissa digits preserve four
significant digits, not arbitrary numerical precision. Exponent errors can
produce large numerical errors. These points require explicit measurement.
