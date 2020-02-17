import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
pd.options.mode.chained_assignment = None

# Loading the training dataset and looking at the top few rows.
train_df = pd.read_json("./train.json")
train_df.head()


test_df = pd.read_json("./test.json")
print("Train Rows :",train_df.shape[0])
print("Test Rows :", test_df.shape[0])

int_level = train_df['interest_level'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[2])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Interest level', fontsize=12)
#plt.show()


cnt_srs = train_df['bathrooms'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('bathrooms', fontsize=12)
#plt.show()

train_df['bathrooms'].ix[train_df['bathrooms']>3] = 3
plt.figure(figsize=(8,4))
sns.violinplot(x="interest_level", y="bathrooms", data=train_df)
plt.xlabel('Interset level', fontsize=12)
plt.ylabel('bathroom', fontsize=12)
#plt.show()

# looks like evently distributed across the interest levels, now let us
# look at the next feature 'bedrooms
cns_srs = train_df['bedrooms'].value_counts()
plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color="salmon")
plt.ylabel('Number of Occurences', fontsize=12)
plt.xlabel('bedrooms', fontsize=12)
#plt.show()

plt.figure(figsize=(8,6))
sns.countplot(x="bedrooms", hue="interest_level",data=train_df)
plt.xlabel('bedrooms',fontsize=18)
plt.ylabel('Number of Occurences', fontsize=18)
#plt.show()

# Now lets looks at the price Distributions
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]),np.sort(train_df.price.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
#plt.show()


# Looks like there are some outliers in this feature => So let us remove them and then plot again
ulimit = np.percentile(train_df.price.values, 99)
train_df['price'].ix[train_df['price']>ulimit] = ulimit
plt.figure(figsize=(8,6))
sns.distplot(train_df.price.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
#plt.show()

# for the features Bedrooms
ulimit = np.percentile(train_df.bedrooms.values,99)
train_df['bedrooms'].ix[train_df['bedrooms']>ulimit] = ulimit
plt.figure(figsize=(8,6))
sns.distplot(train_df.bedrooms.values, bins=50, kde=True)
plt.xlabel('bedrooms', fontsize=12)
plt.ylabel('Occurrences',fontsize=12)

# The distribution is right skewed as we can see
# Now let us look at the latitude and longitude variables
# Latitude & Longitude:

llimit = np.percentile(train_df.latitude.values, 1)
ulimit = np.percentile(train_df.latitude.values, 99)
train_df['latitude'].ix[train_df['latitude']<llimit] = llimit
train_df['latitude'].ix[train_df['latitude']>ulimit] = ulimit

plt.figure(figsize=(8,6))
sns.distplot(train_df.latitude.values, bins=50, kde=False)
plt.xlabel('latitude', fontsize=12)

#so the latitude values are primarily between 40.6 and 40.9.
# Now let us look at the longitude values.
llimit = np.percentile(train_df.longitude.values,1)
ulimit = np.percentile(train_df.longitude.values,99)
train_df['longitude'].ix[train_df['longitude'] < llimit] = llimit


train_df["created"] = pd.to_datetime(train_df["created"])
train_df["date_created"] = train_df["created"].dt.date
cnt_srs = train_df['date_created'].value_counts()


plt.figure(figsize=(12,4))
ax = plt.subplot(111)
ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='vertical')

# Numbers of Photos ~
train_df["num_photos"] = train_df['photos'].apply(len)
cnt_srs = train_df["num_photos"].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.values,cnt_srs.index,alpha=0.8)
plt.xlabel('Number of Photos', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)


train_df['num_photos'].ix[train_df['num_photos'] > 12] = 12
plt.figure(figsize=(12,6))
sns.violinplot(x="num_photos",y="interest_level",data=train_df,order=['low', 'medium','high'])
plt.xlabel('Num of Photos', fontsize=12)
plt.ylabel('Interset Level', fontsize=12)


# Now lets look at the number of feature variable and see its distribution
train_df["num_features"] = train_df['features'].apply(len)
cnt_srs = train_df['num_features'].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.xlabel('Number of Occurences', fontsize=12)
plt.ylabel('Number of features', fontsize=12)

train_df['num_features'].ix[train_df['num_features'] > 17] = 17
plt.figure(figsize=(10,8))
sns.violinplot(y="num_features", x="interest_level", data=train_df, order =['low','medium','high'])
plt.xlabel('Interest Level', fontsize=12)
plt.ylabel('Number of features', fontsize=12)
plt.show()


