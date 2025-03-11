import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeRegressor

from data_extraction import DataExtraction
from filter_ranking import ranking


class AdaRank:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeRegressor(max_depth=3)
            model.fit(X, y, sample_weight=sample_weights)
            predictions = model.predict(X)

            # Evaluate performance using NDCG
            score = ndcg_score([y], [predictions])
            if score <= 0:
                continue

            # Update sample weights based on NDCG score
            loss = 1.0 - score
            sample_weights *= np.exp(self.learning_rate * loss)
            sample_weights /= np.sum(sample_weights)

            # Store model and weight
            self.models.append(model)
            self.weights.append(score)

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for model, weight in zip(self.models, self.weights):
            pred += weight * model.predict(X)
        return pred


if __name__ == '__main__':
    DE = DataExtraction()
    query, parsed_df, df = DE.parsing_df()
    df_ranked = ranking(query, df)
    print(df_ranked)
    documents = df_ranked['long_common_name'].to_list()

    # Step 2: Create TF-IDF embeddings for both documents and query
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents).toarray()
    query_vector = vectorizer.transform([query]).toarray()[0]

    # Step 3: Define initial relevance scores using cosine similarity
    y1 = cosine_similarity(X, query_vector.reshape(1, -1)).flatten()
    y2 = np.array(df_ranked['score'].to_list())

    # Step 4.1: Train AdaRank model with cosine similarity
    model1 = AdaRank(n_estimators=5, learning_rate=0.1)
    model1.fit(X, y1)

    # Step 4.1: Train AdaRank model with the rank implemented by us
    model2 = AdaRank(n_estimators=5, learning_rate=0.1)
    model2.fit(X, y2)

    # Step 5: Predict and rank documents based on the query
    predictions1 = model1.predict(X)
    predictions2 = model2.predict(X)

    # Step 6: Rank documents based on the predicted scores
    ranked_documents1 = sorted(
        zip(documents, predictions1),
        key=lambda x: x[1],
        reverse=True
    )
    ranked_documents2 = sorted(
        zip(documents, predictions2),
        key=lambda x: x[1],
        reverse=True
    )

    # Step 7: Display ranked results
    print("\nRanked Documents with Cosine Similarity:")
    for i, (doc, score) in enumerate(ranked_documents1, 1):
        print(f"{i}. {doc} (Score: {score:.4f})")

    print("\nRanked Documents with our Ranking:")
    for i, (doc, score) in enumerate(ranked_documents2, 1):
        print(f"{i}. {doc} (Score: {score:.4f})")
