import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from imblearn.over_sampling import SMOTE


# 1. Load the data
data = pd.read_csv('data/extracted_features.csv')

# 2. Separate features and target
X = data.drop(columns=['LABEL', 'id'])  # Features
y = data['LABEL']  # Target variable

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Perform PCA
pca = PCA(n_components=0.97)  # Retain 97% of variance
X_pca = pca.fit_transform(X_scaled)

# 5. Select Principal Components
n_components = pca.n_components_

print("Number of principal components selected:", n_components)


X_val = pd.read_csv("data/extracted_test_features.csv")

selected_component_names = []

for i in range(pca.n_components_):
    component_loadings = pca.components_[i]
    top_feature_index = np.argmax(np.abs(component_loadings))
    top_feature_name = X.columns[top_feature_index]
    selected_component_names.append(top_feature_name)

newcsv = "data/extracted_pca.csv"
newtest = "data/extracted_test_pca.csv"
newdf = data.loc[:, ["LABEL"]+selected_component_names]
newdf.to_csv(newcsv, index=False)
X_val = X_val.loc[:, selected_component_names]
X_val.to_csv(newtest, index=False)

#7. Load data
train = pd.read_csv('data/extracted_pca.csv')

#8. Separate features and target
X = train.drop(columns=["LABEL"])
y = train["LABEL"]

#9. Apply SMOTE for balancing classes
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

#10. Combine features and target into one DataFrame
combined_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["LABEL"])], axis=1)

#11. Write combined DataFrame to CSV
combined_df.to_csv('data/combined_pca.csv')