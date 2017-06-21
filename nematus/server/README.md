# Nematus Server
Runs Nematus as a Web Server. Further details tba.

## API
Nematus Server supports several API styles.

### Nematus Translation API

#### Translation Request

`POST http://host:port/translate`

Content-Type: application/json

##### Query Parameters

| Parameter           | Type                  | Description  |
| --------------------|-----------------------| -------------|
| ``segments``        | ``list(list(str))``   | The sentences to be translated (source language). Each sentence is a list of tokens. |
| ``normalize``       | ``boolean``           | Normalise scores by sentence length. Default: ``true``. |
| ``beam_width``      | ``int``               | The beam width to be used for decoding. Default: ``5``. |
| ``character_level`` | ``boolean``           | Enables character- rather than subword-level translation. Default: ``false``. |
| ``n_best``          | ``int``               | Return n best translations per segment. Default: ``1``. |
| ``suppress_unk``    | ``boolean``           | Suppress hypotheses containing UNK. Default: ``false``. |
| ``return_word_alignment`` | ``boolean``     | Return word alignment (source to target language) for each segment. Default: ``false``. |
| ``return_word_probabilities`` | ``boolean`` | Return the probability of each word (target language) for each segment. Default: ``false``. |

Sample request:

```json
{
	"segments": [
		["I", "can", "resist", "everything", "except", "temptation", "."],
		["The", "truth", "is", "rarely", "pure", "and", "never", "simple", "."]
	],
	"return_word_alignment": false,
	"return_word_probabilities": true
}
```

##### Response Body

If successful, the response body contains a JSON object with the following structure:

```json
{
    "status": "ok",
    "data": [
        {
            "translation": ["ich", "kann", "dem", "alles", "außer", "Versuchung", "widerstehen", "."],
            "word_probabilities": [0.7646325826644897, 0.9430494904518127, 0.38647976517677307, 0.26613569259643555, 0.6460939645767212, 0.13612061738967896, 0.8265654444694519, 0.9531617760658264, 0.9988221526145935]
        },
        {
            "translation": ["die", "Wahrheit", "ist", "selten", "rein", "und", "nie", "einfach", "."],
            "word_probabilities": [0.5854464173316956, 0.9951469302177429, 0.9804221987724304, 0.8840637803077698, 0.5384426712989807, 0.9914865493774414, 0.6368535161018372, 0.6743759512901306, 0.988534152507782, 0.9986005425453186]
        }
    ]
}
```

#### Status Request

`GET http://host:port/status`

##### Response Body

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

## Sample Client

A sample client, written in Python, is available in `sample_client.py`.
