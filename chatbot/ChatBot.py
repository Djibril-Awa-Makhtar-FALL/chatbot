import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dictionnaire de questions avec plusieurs réponses possibles
responses = {
    "qu'est-ce que la fonction de répartition ?": [
        "Soit X une variable aléatoire, on appelle fonction de répartition de X, que l’on note F_X, "
        "la fonction définie sur R par : F_X(x) = P(X ≤ x)."
    ],
    "qu'est-ce qu'une quantile d'ordre q ?": [
        "On appelle quantile d’ordre q de la variable X, où q ∈ [0, 1], la valeur xq telle que P(X ≤ xq) = q "
        "ou, de même, FX(xq) = q."
    ],
    "qu'est-ce que la théorie des probabilités ?": [
        """La théorie des probabilités a pour objet l’étude des phénomènes aléatoires<br>
        ou du moins considérés comme tels par l’observateur. Pour cela, on introduit le concept d’expérience <br>
        aléatoire dont l’ensemble des résultats possibles constitue l’ensemble fondamental, noté habituellement Ω."""
    ],
    "qu'est-ce qu'une variable aléatoire ?": [
        """On parle de variable aléatoire (abréviation : v.a.) lorsque les résultats sont numériques, c’est-à-dire<br>
        que Ω est identique à tout ou une partie de l’ensemble des nombres réels R.<br>
        On distingue habituellement :<br>
        - les variables aléatoires discrètes pour lesquelles l’ensemble Ω des résultats possibles est un <br>
        ensemble discret de valeurs numériques x1, x2, ..., xn,<br>
        fini ou infini (typiquement : l’ensemble des entiers naturels) ;<br>
        - les variables aléatoires continues pour lesquelles l’ensemble Ω est tout R<br>
        (ou un intervalle de R ou, plus rarement, une union d’intervalles)."""
    ],
    "qu'est-ce qu'une espérance mathématique ?": [
        """On appelle espérance mathématique de X, si elle existe, la valeur notée E(X) telle que :<br>
        E(X) = Σ (xi * pX(xi)) dans le cas discret,<br>
        E(X) = ∫ (x * fX(x) dx) dans le cas continu.<br>
        Du point de vue du graphe de fX (respectivement pX), cette valeur représente le <br>
        centre de gravité de la distribution."""
    ],
    "qu'est-ce que l'espérance d'une fonction ?": [
        """Soit g(X) une fonction de la v.a. X, alors :<br>
        E(g(X)) = ∫ g(x) fX(x) dx dans le cas continu (si l’intégrale existe),<br>
        E(g(X)) = Σ g(xi) pX(xi) dans le cas discret (si la somme existe).<br>
        On voit donc que pour le calcul de E(g(X)), il suffit de remplacer la valeur de x par g(x) (ou xi par g(xi))."""
    ]
}

def normalize_query(query):
    normalized = query.lower().strip()
    return normalized

def get_most_relevant_sentence(user_query, text_data):
    sentences = ' '.join(text_data.values()).split('. ')
    sentences.append(user_query)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    most_relevant_index = cosine_similarities.argmax()
    return sentences[most_relevant_index]

def chatbot(user_query):
    normalized_query = normalize_query(user_query)

    if "espérance mathématique" in normalized_query:
        return responses["qu'est-ce qu'une espérance mathématique ?"][0]
    elif "espérance" in normalized_query and "fonction" in normalized_query:
        return responses["qu'est-ce que l'espérance d'une fonction ?"][0]
    elif "fonction de répartition" in normalized_query:
        return responses["qu'est-ce que la fonction de répartition ?"][0]
    elif "quantile d'ordre q" in normalized_query:
        return responses["qu'est-ce qu'une quantile d'ordre q ?"][0]
    elif "théorie des probabilités" in normalized_query:
        return responses["qu'est-ce que la théorie des probabilités ?"][0]
    elif "variable aléatoire" in normalized_query:
        return responses["qu'est-ce qu'une variable aléatoire ?"][0]
    else:
        return get_most_relevant_sentence(user_query, responses)

# Interface utilisateur avec Streamlit
st.title("Chatbot Statistique et Probabilités")
user_question = st.text_input("Posez votre question sur la statistique ou les probabilités :")

if user_question:
    response = chatbot(user_question)
    st.markdown(response, unsafe_allow_html=True)  # Utiliser unsafe_allow_html pour permettre les balises HTML