#########################################################################################
Business Problem: Perform hierarchical and K-means clustering on the dataset. After that, 
perform PCA on the dataset and extract the first 3 principal components and 
make a new dataset with these 3 principal components as the columns. 
Now, on this new dataset, perform hierarchical and K-means clustering. 
Compare the results of clustering on the original dataset and clustering on the principal components 
dataset (use the scree plot technique to obtain the optimum number of clusters in 
K-means clustering and check if you’re getting similar results with and without PCA).

#############################################################################################

# Importing all required libraties
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engine.outliers import Winsorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from feature_engine.transformation import YeoJohnsonTransformer
from kneed import KneeLocator

import statsmodels.api as sm
import scipy.stats as stats
import pylab

# Reading the .csv file into Python
df = pd.read_csv('D:/Hands on/10_Demension Reduction - PCA/Assignment/wine.csv')

# Basic information of the Dataset
df.info()

# Statistical calculations
df.describe()

# Prints top 5 records
df.head()

# Prints columns of the Dataset
df.columns

# First moment decession / measure of central tendency
df.mean()

df.median()

df.mode()

# Second moment business decession / measure of dispersion
df.var()
df.std()

# Third moment business decession / measure of symmetry
df.skew()

# Fourth momnet business decession / Measure of peakedness
df.kurt()

# Checking whether df Dataset has any NULL values or not
df.isnull().sum()
df.isna().sum()

# Checking whether df  Dataset has duplicated values or not and store it in df_dup 
df_dup = df.duplicated()

# Print sum of duplicate values
df_dup.sum()

# Print the values and its count
df_dup.value_counts()

# Print the Unique values
print(df_dup.unique())

# Print the number of Unique values
print(df_dup.nunique())

# Variance of the Dataset
df.var()

# Correlation coefficient 
df.corr()

# Scatter plots on multiple columns / Pairplot
sns.pairplot(df)

# Multiple Boxplot
df.plot(kind = 'box', subplots = 1, sharey = 0, figsize = (10, 6))
plt.subplots_adjust(wspace = 0.75)

# Alternative to create Multiple Boxplots
for i in df.columns:
    plt.boxplot(df[i])
    plt.title('Box plot for ' + str(i))
    plt.show()


# Applying Winsorization on features which are having outliers
malic_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Malic'])
df.Malic = malic_winsor.fit_transform(df[['Malic']])

ash_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Ash'])
df.Malic = ash_winsor.fit_transform(df[['Ash']])

Alcalinity_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Alcalinity'])
df.Alcalinity = Alcalinity_winsor.fit_transform(df[['Alcalinity']])

Magnesium_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Magnesium'])
df.Magnesium = Magnesium_winsor.fit_transform(df[['Magnesium']])

Pro_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Proanthocyanins'])
df.Proanthocyanins = Pro_winsor.fit_transform(df[['Proanthocyanins']])

Color_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Color'])
df.Color = Color_winsor.fit_transform(df[['Color']])

Hue_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Hue'])
df.Hue = Hue_winsor.fit_transform(df[['Hue']])

ash_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['Ash'])
df.Malic = ash_winsor.fit_transform(df[['Ash']])

# Boxplots after applying Winsorization
for i in df.columns:
    plt.boxplot(df[i])
    plt.title('Box plot for ' + str(i))
    plt.show()
    
# Normal QQ plot on all the features
for i in df.columns[:]:
    stats.probplot(df[i], dist = 'norm', plot = pylab)
    plt.show()

# Alternative approch to plot Normal QQ plot
for i in df.columns[:]:
    sm.qqplot(df[i], line ='45')
    plt.show()

# Standardization
stnd = StandardScaler()
df_stand = pd.DataFrame(stnd.fit_transform(df))
#r_stnd = RobustScaler()
#df_stand = pd.DataFrame(stnd.fit_transform(df))
#norm = MinMaxScaler()
#df_stand = pd.DataFrame(r_stnd.fit_transform(df))

# Statistical calculations on Standardized data
df_stand.describe()

# Mean
df_stand.mean()

# Standard deviation
df_stand.std()

# Variance 
df_stand.var()

# Dendrogram
tree_plot = dendrogram(linkage(df_stand, method = 'ward'))

# Creating AgglomerativeClustering object and predicting output labels
hc1 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc1 = hc1.fit_predict(df_stand)

# Silhouette score to determine the accuracy
silhouette_score(df_stand, y_hc1)

# Creating KMeans object and predicting output labels
kmeans = KMeans(n_clusters = 3)
model = kmeans.fit(df_stand)

# Silhouette score to determine the accuracy
silhouette_score(df_stand, model.labels_)

# Creaing PCA object and Executing
pca = PCA(n_components = 14)
pca_model = pca.fit(df_stand)

# Transforming the values
trans_pca = pd.DataFrame(pca_model.transform(df_stand))

# Creating a DataFrame obejct with PCA values and label them with respective names
components = pd.DataFrame(pca_model.components_, columns = df.columns).T
components.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11', 'pc12', 'pc13', 'pc14']
components

# PCA values of the each PC's
pca_model.explained_variance_ratio_

# Cumulative percent of PCA values
sum = np.cumsum(pca_model.explained_variance_ratio_)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(var1, c = 'red')

# Creating Knee locator diagram
kl = KneeLocator(range(len(var1)), var1, curve = 'concave', direction = "increasing") 
# The line is pretty linear hence Kneelocator is not able to detect the knee/elbow appropriately
kl.elbow
plt.style.use("seaborn")
plt.plot(range(len(var1)), var1)
plt.xticks(range(len(var1)))
plt.ylabel("Interia")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show()

# Taking first three columns the data set
df_pca = trans_pca.iloc[:, 0:3]
df_pca.columns = ['pc1', 'pc2', 'pc3']

# Statistical calculations
df_pca.describe()

# Standard Deviation
df_pca.std()

# Creating AgglomerativeClustering object and predicting the labels
hc1 = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc1 = hc1.fit_predict(df_stand)

# Silhouette score to determine the accuracy
silhouette_score(df_stand, y_hc1)

# Creating KMeans object and predicting the labels
kmeans = KMeans(n_clusters = 3)
model = kmeans.fit_predict(df_stand)

# Silhouette score to determine the accuracy
silhouette_score(df_stand, model)

# Tried various approaches to choose the best value to cluster
'''
k = []
for i in range(2, 10):
    hc1 = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage = 'ward')
    y_hc1 = hc1.fit_predict(pca_stand)
    
    k.append(silhouette_score(pca_stand, y_hc1))
    
print(k)


l = []
for i in range(3, 15):
    kmeans = KMeans(n_clusters = i)
    model = kmeans.fit(pca_stand)
    l.append(silhouette_score(pca_stand, model.labels_))
    
print(l)'''

################################################################################################
Business Problem: A pharmaceuticals manufacturing company is conducting a study on a new medicine 
to treatheart diseases. The company has gathered data from its secondary sources and would like 
you to provide high level analytical insights on the data. Its aim is to segregate patients 
depending on their age group and other factors given in the data. Perform PCA and 
clustering algorithms on the dataset and check if the clusters formed before and 
after PCA are the same and provide a brief report on your model. You can also explore more ways 
to improve your model. 
################################################################################################

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab

from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Reading .csv file into Python
df = pd.read_csv('D:/Hands on/10_Demension Reduction - PCA/Assignment/heart disease.csv')

# Information of the Dataset
df.info()

# Statistical calculations
df.describe()

# Print top five records
df.head()

# Datatype of the dataset
df.dtypes

# Columns of the dataset
df.columns

# First moment decession / measure of central tendency
df.mean()

df.median()

df.mode()

# Second moment business decession / measure of dispersion
df.var()

df.std()

# Third moment business decession / measure of symmetry
df.skew()

# Fourth momnet business decession / Measure of peakedness
df.kurt()


# Pairplot
sns.pairplot(df)

# Checking whether Dataset has NULL values
df.isna().sum()
df.isnull().sum()

# Checking wheter Dataset has duplicated values
df_dup = df.duplicated()

# Sum of duplicate values
df_dup.sum()

# Print unique values and its count
df_dup.value_counts()

# Print number of unique values
df_dup.nunique()

# Deleting duplicate values and store it into df1
df1 = df.drop_duplicates()

# for loop to plot boxplot for all the columns
for i in df1.columns[:]:
    plt.boxplot(df1[i])
    plt.title('Box plot for ' + str(i))
    plt.show()

# Applying Winsorization technique on features which are having outliers
trestbps_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['trestbps'])
df1['trestbps'] = trestbps_winsor.fit_transform(df1[['trestbps']])

chol_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['chol'])
df1['chol'] = chol_winsor.fit_transform(df1[['chol']])

fbs_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['fbs'])
df1['fbs'] = fbs_winsor.fit_transform(df1[['fbs']])

thalach_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['thalach'])
df1['thalach'] = thalach_winsor.fit_transform(df1[['thalach']])

oldpeak_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['oldpeak'])
df1['oldpeak'] = oldpeak_winsor.fit_transform(df1[['oldpeak']])

ca_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['ca'])
df1['ca'] = ca_winsor.fit_transform(df1[['ca']])

thal_winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['thal'])
df1['thal'] = thal_winsor.fit_transform(df1[['thal']])

# for loop to plot boxplots after Winsorization
for i in df1.columns[:]:
    plt.boxplot(df1[i])
    plt.title('Box plot for ' + str(i))
    plt.show()
    

# Normal QQ plot
for i in df1.columns:
    stats.probplot(df[i], dist = 'norm', plot = pylab)
    plt.title('QQ plot for '+ str(i))
    plt.show()

# Appyling RobustScaler technique
rob_scale = RobustScaler()
df_standrd = pd.DataFrame(rob_scale.fit_transform(df1))

'''
std_scale = StandardScaler()
min_scale = MinMaxScaler()
df_standrd = std_scale.fit_transform(df1)    
df_standrd = min_scale.fit_transform(df1)
'''

# Standard deviation
df_standrd.std()

# Mean
df_standrd.mean()
    
# Dendrogram
tree_plot = dendrogram(linkage(df_standrd, method = 'ward'))

# Creating AgglomerativeClustering object and predicting output labels
hc1 = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'single')
y_hc1 = hc1.fit_predict(df_standrd)

# Silhouette score to determine the accuracy of the model
silhouette_score(df_standrd, y_hc1)   

# Creating KMeans object and executing
kmeans = KMeans(n_clusters = 2)
model = kmeans.fit_predict(df_standrd)

# Silhouette score to determine the accuracy of the model    
silhouette_score(df_standrd, model)

# Creating Cluster_lable columns    
df1['Cluster_label'] = pd.DataFrame(model)

# Creating diff Dataset for diff clusters
cluster0 = df1.loc[df1.Cluster_label == 0]
cluster1 = df1.loc[df1.Cluster_label == 1]

# Creating PCA object and executing
pca = PCA(n_components = 14)
pca_model = pca.fit(df_standrd)

# Transforming the values
trans_pca = pd.DataFrame(pca_model.transform(df_standrd))

# Creating a Dataset with PCA values
pca_df = pd.DataFrame(pca_model.components_, columns = df_standrd.columns)
pca_df.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10', 'pc11', 'pc12', 'pc13', 'pc14', ]

# PCA percentage of each PCA
var = pca_model.explained_variance_ratio_

# Cumulative percentage 
var1 = np.cumsum(var)

# Taking first 10 columns
pca_new_data = trans_pca.iloc[:, 0:10]

# Dedrogram
tree_plot_after = dendrogram(linkage(pca_new_data, method = 'ward'))

# Creating AgglomerativeClustering object and predicting output labels
hc1 = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'single')
y_hc1 = hc1.fit_predict(pca_new_data)

# Silhouette score to determine the accuracy of the model    
silhouette_score(pca_new_data, y_hc1)

# Creating KMeans object and executing    
kmeans = KMeans(n_clusters = 2)
model = kmeans.fit_predict(pca_new_data)
    
# Silhouette score to determine the accuracy of the model    
silhouette_score(pca_new_data, model)

# creating a new feature in df1 Dataset
df1['Cluster_label'] = pd.DataFrame(model)

# Creating diff Datasets to the diff clusters
cluster0 = df1.loc[df1.Cluster_label == 0]
cluster1 = df1.loc[df1.Cluster_label == 1]



