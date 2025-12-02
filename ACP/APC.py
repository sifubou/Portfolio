import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# 1. ACP Linéaire (Sections 3-6)
# -------------------------------------------------------------------

def fit_pca(X, n_components=None):
    """
    Implémentation de l'ACP linéaire selon les sections 3-6 du document.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Matrice de données brutes
    n_components : int, optional
        Nombre de composantes principales à conserver
    
    Returns:
    --------
    components : array, shape (n_features, n_components)
        Vecteurs propres (composantes principales)
    mean : array, shape (n_features,)
        Moyenne des données
    eigenvalues : array, shape (n_features,)
        Valeurs propres de la matrice de covariance
    """
    # Conversion en array numpy
    X = np.array(X)
    n, p = X.shape
    
    if n_components is None:
        n_components = p
    
    # Section 2.3 : Centrage des données
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # Section 3.2 : Matrice de covariance (formule du document : 1/n)
    Sigma = (X_centered.T @ X_centered) / n
    
    # Section 6.4 : Résolution du problème aux valeurs propres
    # Utilisation de eigh pour matrices symétriques (plus stable)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    # Tri décroissant des valeurs propres et vecteurs propres
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Sélection des n_components premières composantes
    components = eigenvectors[:, :n_components]
    
    return components, mean, eigenvalues

def transform_pca(X, components, mean):
    """
    Projection des données sur les composantes principales.
    
    Section 4.1 : z = Xw
    """
    X_centered = X - mean
    return X_centered @ components

def inverse_transform_pca(X_proj, components, mean):
    """
 Reconstruction des données à partir des composantes principales.
    """
    return X_proj @ components.T + mean

# -------------------------------------------------------------------
# 2. ACP à Noyau (Sections 10)
# -------------------------------------------------------------------

def rbf_kernel(X1, X2, gamma=1.0):
    """
    Noyau RBF (Gaussien) selon la section 10.1
    """
    X1 = np.array(X1)
    X2 = np.array(X2)
    
    # Calcul des distances au carré ||x - y||²
    X1_norm = np.sum(X1**2, axis=1, keepdims=True)
    X2_norm = np.sum(X2**2, axis=1, keepdims=True)
    distances = X1_norm + X2_norm.T - 2 * X1 @ X2.T
    
    return np.exp(-gamma * distances)

def center_kernel_matrix(K):
    """
    Centrage de la matrice noyau selon la section 10.2
    """
    n = K.shape[0]
    ones_n = np.ones((n, n)) / n
    K_centered = K - ones_n @ K - K @ ones_n + ones_n @ K @ ones_n
    return K_centered

def fit_kpca(X, n_components=None, kernel='rbf', gamma=1.0):
    """
    Implémentation de l'ACP à noyau selon la section 10 du document.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
    n_components : int, optional
    kernel : str, kernel à utiliser
    gamma : float, paramètre du noyau RBF
    
    Returns:
    --------
    model : dict
        Modèle contenant tous les paramètres nécessaires
    """
    X = np.array(X)
    n, p = X.shape
    
    if n_components is None:
        n_components = n
    
    # Section 10.1 : Matrice noyau
    if kernel == 'rbf':
        K = rbf_kernel(X, X, gamma)
    else:
        raise ValueError("Seul le noyau RBF est implémenté")
    
    # Section 10.2 : Centrage de la matrice noyau
    K_centered = center_kernel_matrix(K)
    
    # Section 10.6 : Problème aux valeurs propres
    # K_centered * alpha = n * lambda * alpha
    eigenvalues, alphas = np.linalg.eigh(K_centered)
    
    # Tri décroissant
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    alphas = alphas[:, idx]
    
    # Filtrage des valeurs propres négatives (erreurs numériques)
    mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[mask]
    alphas = alphas[:, mask]
    
    # Section 10.7 : Normalisation
    # On normalise pour que alpha^T * K * alpha = 1
    # Ce qui équivaut à alpha_normalized = alpha / sqrt(lambda * n)
    alphas_normalized = alphas / np.sqrt(eigenvalues * n).reshape(1, -1)
    
    # Sélection des composantes
    alphas_k = alphas_normalized[:, :min(n_components, alphas_normalized.shape[1])]
    eigenvalues_k = eigenvalues[:min(n_components, len(eigenvalues))]
    
    # Stockage du modèle
    model = {
        'alphas': alphas_k,
        'eigenvalues': eigenvalues_k,
        'X_train': X,
        'kernel': kernel,
        'gamma': gamma,
        'K_train': K,  # Pour le centrage des nouvelles données
        'n_components': alphas_k.shape[1]
    }
    
    return model

def transform_kpca(X, model):
    """
    Projection de nouvelles données selon la section 10.8
    """
    X = np.array(X)
    
    # Extraction des paramètres du modèle
    alphas = model['alphas']
    X_train = model['X_train']
    kernel = model['kernel']
    gamma = model['gamma']
    K_train = model['K_train']
    
    # Matrice noyau entre nouvelles données et données d'entraînement
    if kernel == 'rbf':
        K_test = rbf_kernel(X, X_train, gamma)
    else:
        raise ValueError("Seul le noyau RBF est implémenté")
    
    # Centrage de K_test
    n_train = K_train.shape[0]
    n_test = K_test.shape[0]
    
    ones_n_train = np.ones((n_train, n_train)) / n_train
    ones_n_test_train = np.ones((n_test, n_train)) / n_train
    
    K_test_centered = (K_test 
                       - ones_n_test_train @ K_train 
                       - K_test @ ones_n_train 
                       + ones_n_test_train @ K_train @ ones_n_train)
    
    # Projection : z = K_test_centered * alpha
    return K_test_centered @ alphas

# -------------------------------------------------------------------
# 3. Fonctions d'analyse et de visualisation
# -------------------------------------------------------------------

def explained_variance_ratio(eigenvalues):
    """
    Calcul des proportions de variance expliquée
    Section 7.5
    """
    total_variance = np.sum(eigenvalues)
    return eigenvalues / total_variance

def plot_pca_results(X, X_pca, y, title):
    """
    Visualisation des résultats de l'ACP
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('Données originales')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('Composante Principale 1')
    plt.ylabel('Composante Principale 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_variance_explained(eigenvalues, title):
    """
    Graphique de la variance expliquée
    """
    explained_variance = explained_variance_ratio(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
    plt.title(f'{title} - Variance Expliquée')
    plt.xlabel('Composante Principale')
    plt.ylabel('Proportion de Variance')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
    plt.title(f'{title} - Variance Cumulée')
    plt.xlabel('Nombre de Composantes')
    plt.ylabel('Variance Cumulée')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 4. Exemples démonstratifs
# -------------------------------------------------------------------

def example_linear_data():
    """
    Exemple avec des données linéairement séparables
    """
    print("=== EXEMPLE DONNÉES LINÉAIRES ===")
    
    # Génération de données
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    X = np.dot(X, [[2, 1], [1, 2]])  # Transformation linéaire
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # ACP Linéaire
    components, mean, eigenvalues = fit_pca(X, n_components=2)
    X_pca = transform_pca(X, components, mean)
    
    # Analyse
    explained_ratio = explained_variance_ratio(eigenvalues)
    print(f"Valeurs propres: {eigenvalues}")
    print(f"Variance expliquée: {explained_ratio}")
    print(f"Variance cumulée: {np.cumsum(explained_ratio)}")
    
    # Visualisation
    plot_pca_results(X, X_pca, y, "ACP Linéaire - Données Linéaires")
    plot_variance_explained(eigenvalues, "ACP Linéaire")

def example_nonlinear_data():
    """
    Exemple avec des données non-linéaires (lunes)
    """
    print("\n=== EXEMPLE DONNÉES NON-LINÉAIRES ===")
    
    # Génération de données non-linéaires
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ACP Linéaire
    components_lin, mean_lin, eigenvalues_lin = fit_pca(X_scaled, n_components=2)
    X_pca_lin = transform_pca(X_scaled, components_lin, mean_lin)
    
    # ACP à Noyau
    model_kpca = fit_kpca(X_scaled, n_components=2, kernel='rbf', gamma=15)
    X_kpca = transform_kpca(X_scaled, model_kpca)
    
    # Visualisation comparative
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('Données Originales (Lunes)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca_lin[:, 0], X_pca_lin[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('ACP Linéaire (Échec)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('ACP à Noyau (Succès)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("L'ACP linéaire ne peut pas séparer les données non-linéaires")
    print("L'ACP à noyau réussit à linéariser la structure")

def example_circles():
    """
    Exemple avec des cercles concentriques
    """
    print("\n=== EXEMPLE CERCLES CONCENTRIQUES ===")
    
    X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    # ACP Linéaire
    components_lin, mean_lin, eigenvalues_lin = fit_pca(X_scaled, n_components=2)
    X_pca_lin = transform_pca(X_scaled, components_lin, mean_lin)
    
    # ACP à Noyau
    model_kpca = fit_kpca(X_scaled, n_components=2, kernel='rbf', gamma=10)
    X_kpca = transform_kpca(X_scaled, model_kpca)
    
    # Visualisation
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('Données Originales (Cercles)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca_lin[:, 0], X_pca_lin[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('ACP Linéaire')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title('ACP à Noyau')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 5. Test avec l'exemple numérique du document (Section 7)
# -------------------------------------------------------------------

def test_document_example():
    """
    Reproduction de l'exemple numérique de la section 7 du document
    """
    print("=== EXEMPLE NUMÉRIQUE DU DOCUMENT (SECTION 7) ===")
    
    # Données brutes de l'exemple
    X_brut = np.array([
        [0, 0],
        [1, 1], 
        [2, 1],
        [3, 4]
    ])
    
    print("Données brutes:")
    print(X_brut)
    
    # Calcul manuel pour vérification
    mean_manual = np.mean(X_brut, axis=0)
    X_centered_manual = X_brut - mean_manual
    Sigma_manual = (X_centered_manual.T @ X_centered_manual) / 4
    
    print(f"\nMoyennes calculées: {mean_manual}")
    print("Matrice de covariance calculée:")
    print(Sigma_manual)
    
    # ACP avec notre implémentation
    components, mean, eigenvalues = fit_pca(X_brut, n_components=2)
    X_pca = transform_pca(X_brut, components, mean)
    
    print(f"\nValeurs propres: {eigenvalues}")
    print("Vecteurs propres (composantes):")
    print(components)
    print("\nDonnées projetées:")
    print(X_pca)
    
    # Vérification de la variance expliquée
    explained_ratio = explained_variance_ratio(eigenvalues)
    print(f"\nVariance expliquée: {explained_ratio}")
    print(f"Variance cumulée: {np.cumsum(explained_ratio)}")

# -------------------------------------------------------------------
# 6. Exécution des exemples
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Test de l'exemple du document
    test_document_example()
    
    # Exemples démonstratifs
    example_linear_data()
    example_nonlinear_data() 
    example_circles()
    
    print("\n=== SYNTHÈSE ===")
    print("✓ ACP Linéaire implémentée selon les sections 3-6")
    print("✓ ACP à Noyau implémentée selon la section 10") 
    print("✓ Exemple numérique du document reproduit")
    print("✓ Visualisations comparatives générées")