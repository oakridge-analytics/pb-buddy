## Bike Buddy API

### Usage:

- Copy Autgluon MultiModal model snapshot folder, and paired sklearn Pipeline to transform input ad data -> prediction input dataframe expected
    - These are generated in `reports/modelling/`
    - Load to S3, under `s3://bike-buddy/models`
    - Update `main.py` with the correct S3 path to the model snapshot folder and the paired sklearn Pipeline
- Update modal image build steps to force build if needing an update (re-download model into image, pip install from repo)

- Serve as ephemeral API with `modal`:

```console
modal serve main.py
```

- Then test with:

```console
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
curl -X POST -H "Content-Type: application/json" -d @test.json  https://dbandrews--bike-buddy-api-autogluonmodelinference-predict-dev.modal.run
```

- To deploy once happy:

```console
modal deploy main.py
```

