# Nematus Server
Runs Nematus as a web service.

## Basic Usage

The command

```bash
python server.py -m model.npz
```

will start Nematus Server at `localhost` on port 8080, using translation model `model.npz`. Once the model has been loaded, the server is ready to answer translation requests according to the API outlined below.

### Required Arguments

Nematus Server needs at least one translation model, provided via the `-m` or `--models` parameter. Multiple models (for ensemble decoding) are delimited with spaces:

```bash
python server.py -m model1.npz model2.npz model3.npz model4.npz
```

### Optional Arguments

| Argument            | Default Value | Description              |
| --------------------|---------------| -------------------------|
| `--host`            | `localhost`   | Host name                |
| `--port`            | `8080`        | Port                     |
| `-p`,               | `1`           | Number of translation processes to start. Each process loads all models specified in `-m`/`--models`. |
| `--device-list`     | any           | The devices to start translation processes on, e.g., `gpu0 gpu1 gpu6`. Defaults to any available device. |
| `-v`                | off           | Verbose mode             |


## API
Nematus Server supports several API styles.

### Nematus Translation API

#### Translation Request

`POST http://host:port/translate`

Content-Type: application/json

##### Query Parameters

| Parameter           | Type                  | Default Value | Description |
| --------------------|-----------------------|-----------|-------------|
| ``segments``        | ``list(list(str))``   |           | The sentences to be translated (source language). Each sentence is a list of tokens. |
| ``normalize``       | ``boolean``           | ``true``  | Normalise scores by sentence length. |
| ``beam_width``      | ``int``               | ``5``     | The beam width to be used for decoding. |
| ``character_level`` | ``boolean``           | ``false`` | Enables character- rather than subword-level translation. |
| ``n_best``          | ``int``               | ``1``     | Return n best translations per segment. |
| ``suppress_unk``    | ``boolean``           | ``false`` | Suppress hypotheses containing UNK. |
| ``return_word_alignment`` | ``boolean``     | ``false`` | Return word alignment (source to target language) for each segment. |
| ``return_word_probabilities`` | ``boolean`` | ``false`` | Return the probability of each word (target language) for each segment. |

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
