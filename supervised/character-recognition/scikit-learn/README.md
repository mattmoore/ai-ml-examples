# Using scikit to detect digits from handwritten images

## Publish the model to Google Cloud ML

```shell
gcloud ml-engine versions create v1 --model digit_recognition --origin gs://digit_recognition/ --runtime-version 1.12 --python-version 3.5
```

Run the published version with the JSON-formatted number 2 digit:

```shell
gcloud ml-engine predict --model digit_recognition --version v1 --json-instances digit_2.json
```
