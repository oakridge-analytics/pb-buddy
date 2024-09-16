# Bike Buddy App

To demo the Bike Buddy price prediction model, a simple interface for parsing ads and getting predicted price.

### Usage:

- Serve as ephemeral app with `modal` from root of repo:

```console
modal serve app/main.py
```

- To deploy once happy:

```console
modal deploy app/main.py
```

To run locally, install app specific dependencies into `pb-buddy` environment with `pip` then run:

```console
pip install -r app/requirements.txt
python app/app.py
```

