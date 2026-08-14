import streamlit as st
import joblib
import numpy as np


st.set_page_config(
    page_title="News Bias Indicator",
    page_icon="📰",
    layout="centered"
)


@st.cache_resource
def load_model():
    model = joblib.load("bias_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()


st.title("📰 News Bias Indicator")

st.write(
    "Analyze a news article and estimate its "
    "ideological leaning as Left, Neutral, or Right."
)

st.divider()


article = st.text_area(
    "📝 Enter your news article",
    height=250,
    placeholder="Paste your news article here..."
)


if st.button("🔍 Analyze Article", use_container_width=True):

    if article.strip() == "":
        st.warning("⚠️ Please enter a news article first.")
    else:
        article_tfidf = vectorizer.transform([article])

        prediction = model.predict(article_tfidf)[0]

        st.subheader("📊 Predicted Bias")

        if prediction == "Left":
            st.error("🔴 LEFT")
        elif prediction == "Right":
            st.warning("🔵 RIGHT")
        elif prediction == "Neutral":
            st.success("🟢 NEUTRAL")
        else:
            st.info(f"Prediction: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(article_tfidf)[0]
            classes = model.classes_

            st.subheader("📈 Confidence Levels")

            confidence_scores = {}
            for class_name, probability in zip(classes, probabilities):
                confidence_scores[class_name] = probability * 100

            if "Left" in confidence_scores:
                left_score = confidence_scores["Left"]
                st.write(f"🔴 **Left: {left_score:.2f}%**")
                st.progress(min(int(left_score), 100))

            if "Neutral" in confidence_scores:
                neutral_score = confidence_scores["Neutral"]
                st.write(f"🟢 **Neutral: {neutral_score:.2f}%**")
                st.progress(min(int(neutral_score), 100))

            if "Right" in confidence_scores:
                right_score = confidence_scores["Right"]
                st.write(f"🔵 **Right: {right_score:.2f}%**")
                st.progress(min(int(right_score), 100))

        st.subheader("🔎 Supporting Words")

        try:
            feature_names = vectorizer.get_feature_names_out()
            tfidf_values = article_tfidf.toarray()[0]
            coefficients = model.coef_
            classes = model.classes_

            class_index = list(classes).index(prediction)
            class_coefficients = coefficients[class_index]

            contributions = np.zeros_like(tfidf_values)
            for idx in range(len(tfidf_values)):
                if tfidf_values[idx] > 0:
                    contributions[idx] = tfidf_values[idx] * class_coefficients[idx]

            top_indices = np.argsort(contributions)[::-1]

            count = 0
            for index in top_indices:
                if contributions[index] > 0:
                    word = feature_names[index]
                    st.write(f"• **{word}**")
                    count += 1

                if count == 4:
                    break

            if count == 0:
                st.write("No strong supporting words were identified from the article.")

        except Exception:
            st.write("Supporting-word explanation is not available for this model.")

        st.subheader("💡 Interpretation")
        st.write(
            f"The model classified this article as "
            f"**{prediction}** based on patterns "
            "learned from the training dataset."
        )

        st.divider()
        st.subheader("⚠️ Disclaimer")
        st.info(
            "The confidence levels represent the "
            "model's estimated probabilities, not "
            "absolute certainty. Supporting words "
            "are model-derived indicators based on "
            "learned features present in the text. "
            "They should not be treated as proof that "
            "an article is objectively Left, Right, or Neutral."
        )