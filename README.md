# News Bias Indicator

A Streamlit application that uses a trained machine-learning model to estimate
whether a news article is **Left**, **Neutral**, or **Right**. It can also show
the model's estimated confidence by class and supporting words when the model
artifacts provide the required information.

## Project files

- `app.py` — Streamlit user interface and prediction logic.
- `bias_model.pkl` — trained classification model.
- `tfidf_vectorizer.pkl` — fitted TF-IDF text vectorizer used by the model.
- `requirements.txt` — Python package dependencies.

Keep both `.pkl` files in the same directory as `app.py`; the application loads
them by filename when it starts.

## Run locally

1. Install Python.
2. From this project directory, install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL printed by Streamlit, paste an article into the text
   box, and select **Analyze Article**.

## Important note

This tool provides a machine-learning estimate, not an objective or definitive
judgment of an article's political bias. Confidence values are model-estimated
probabilities, and supporting words are indicators derived from the model.
Use the results with care.