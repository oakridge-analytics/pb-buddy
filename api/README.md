## Bike Buddy API

Usage:
- Copy MultiModal model snapshot, and paired sklearn Pipeline to transform input ad data -> prediction input dataframe expected
- Update .env file, pointing to correct files
- Setup environment with from within `api` folder:

```
conda create --name pb-buddy-api python=3.9
conda activate pb-buddy-api
pip install -r requirements.txt
cd .. && pip install -e .
```
- Use `source .env` to set env variables in current shell
- Launch server with:

```
uvicorn main:app --port 8000
```

- Test with:

```
curl -X POST -H "Content-Type: application/json" -d @test.json http://localhost:8000/text-predict
```