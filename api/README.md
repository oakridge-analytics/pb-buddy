## Bike Buddy API

### Usage:

- Copy MultiModal model snapshot, and paired sklearn Pipeline to transform input ad data -> prediction input dataframe expected
    - These are generated in `reports/modelling/`
- Update .env file, pointing to correct file for sklearn transformer file, and AutoGluon model folder
- Setup environment with from within `api` folder:

```
conda create --name bike-buddy-api python=3.9 -y
conda activate bike-buddy-api
pip install -r requirements.txt
cd .. && pip install -e .
```
- Launch server with:

```
uvicorn main:app --port 8000
```

- Test with:

```
curl -X POST -H "Content-Type: application/json" -d @test.json http://localhost:8000/text-predict
```

### Dockerfile:

Once you have added the model assets to `api/assets` and updated the `.env` file in `api/.env` to point to these files, build the Docker image from the root of the `pb-buddy` repo:

```
docker build -t bikebuddy-api -f api/Dockerfile .
```

Then run the image locally using:

```
docker run -p 8000:8000 -it bikebuddy-api
```

And test with commands above.

#### Azure Deployment Notes

With existing `pbbuddy` Azure Container Registry:

`az acr login --name pbbuddy`

*Optional: Create a new version X.X.X tag (to be improved in a build system....):*

`docker tag bikebuddy-api:latest bikebuddy-api:X.X.X`

`docker tag bikebuddy-api:latest pbbuddy.azurecr.io/bikebuddy:latest`

`docker push pbbuddy.azurecr.io/bikebuddy:latest`


