

Note: Please set `OPEN_AI_KEY` in your environment variable to run this.

Install dependencies:

```python
pip install -r requirements.txt
```

Create the Chroma DB:

```python
python create_database.py
```

Query the Chroma DB:

```python
python query_data.py "What is Andrew Huberman's sleep cocktail?"
```