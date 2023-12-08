from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

# Load the UCI Heart Disease dataset
heart_disease = fetch_openml(name='heart')

# Access data
X = heart_disease.data

# Imputation to handle NaNs
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert to a sparse matrix if the data is not already sparse
if not isinstance(X_imputed, csr_matrix):
    X_imputed = csr_matrix(X_imputed)

# Normalizing the data (for dense or sparse matrices)
scaler = StandardScaler(with_mean=False)  # Specify with_mean=False for sparse matrices
X_scaled = scaler.fit_transform(X_imputed)

# Applying K-Means
kmeans = KMeans(n_clusters=2, random_state=42)  # Define o número de clusters
kmeans.fit(X_scaled)

# Getting cluster labels and calculating silhouette score
labels = kmeans.labels_
silhouette_avg = silhouette_score(X_scaled, labels)
print(f'Coeficiente de silhueta médio: {silhouette_avg}')
