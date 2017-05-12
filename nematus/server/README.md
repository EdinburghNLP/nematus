# Nematus Server
Runs Nematus as a Web Server. Further details tba.

# API
Nematus Server supports several API styles.

## Nematus Translation API

### Translation Request

`POST http://host:port/translate`

Content-Type: application/json

#### Query Parameters

| Parameter           | Type                  | Description  |
| --------------------|-----------------------| -------------|
| ``segments``        | ``array(str)``        | The sentences to be translated (source language). |
| ``normalize``       | ``boolean``           | Normalise scores by sentence length. Default: ``true``. |
| ``beam_width``      | ``int``               | The beam width to be used for decoding. Default: ``5``. |
| ``character_level`` | ``boolean``           | Enables character- rather than subword-level translation. Default: ``false``. |
| ``n_best``          | ``int``               | Return n best translations per segment. Default: ``1``. |
| ``suppress_unk``    | ``boolean``           | Suppress hypotheses containing UNK. Default: ``false``. |
| ``return_word_alignment`` | ``boolean``     | Return word alignment (source to target language) for each segment. Default: ``false``. |
| ``return_word_probabilities`` | ``boolean`` | Return the probability of each word (target language) for each segment. Default: ``false``. |

#### Response Body

If successful, the response body contains a JSON object with the following structure:

```json
{
  "status": "ok",
  "data": [
    {
      "translation": "Translation of first segment."
    },
    {
      "translation": "Translation of second segment."
    }
  ]
}
```

Additional fields for each segment in ``data`` are ``word_alignment`` and ``word_probabilities`` (details tba).

### Status Request

`GET http://host:port/status`

#### Response Body

If successful, the response body contains a JSON object with the following structure:

```json
{
  "status": "ok",
  "models": [
    "wmt16-en-de-model1.npz",
    "wmt16-en-de-model2.npz",
    "wmt16-en-de-model3.npz",
    "wmt16-en-de-model4.npz",
  ],
  "version": "0.1.dev0",
  "service": "nematus"
}
```
