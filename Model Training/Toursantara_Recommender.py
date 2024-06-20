import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

# Load the preprocessed datasets
df = pd.read_csv(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\ML_Repo\Modified Datasets\tourism_rating_modified.csv")
tourism_new = pd.read_csv(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\ML_Repo\Modified Datasets\tourism_new.csv")

# Extract unique user and place information
user_ids = df['User_Id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

place_ids = df['Place_Id'].unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

df['user'] = df['User_Id'].map(user_to_user_encoded)
df['place'] = df['Place_Id'].map(place_to_place_encoded)

num_users = len(user_to_user_encoded)
num_place = len(place_to_place_encoded)
num_cities = df['City_Encoded'].nunique()
num_categories = df['Category_Encoded'].nunique()

# Normalize ratings
df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)
min_rating = min(df['Place_Ratings'])
max_rating = max(df['Place_Ratings'])

# Shuffle and split data
df = df.sample(frac=1, random_state=42)
x = df[['user', 'place', 'City_Encoded', 'Category_Encoded', 'Normalized_Price']].values
y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:]

# Define the RecommenderNet model
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_place, num_cities, num_categories, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.place_embedding = layers.Embedding(num_place, embedding_size,
                                                embeddings_initializer='he_normal',
                                                embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.place_bias = layers.Embedding(num_place, 1)
        self.city_embedding = layers.Embedding(num_cities, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.category_embedding = layers.Embedding(num_categories, embedding_size,
                                                   embeddings_initializer='he_normal',
                                                   embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.price_dense = layers.Dense(embedding_size)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        place_vector = self.place_embedding(inputs[:, 1])
        place_bias = self.place_bias(inputs[:, 1])
        city_vector = self.city_embedding(inputs[:, 2])
        category_vector = self.category_embedding(inputs[:, 3])
        price_vector = self.price_dense(tf.expand_dims(inputs[:, 4], axis=-1))

        dot_user_place = tf.tensordot(user_vector, place_vector, 2)
        dot_user_city = tf.tensordot(user_vector, city_vector, 2)
        dot_user_category = tf.tensordot(user_vector, category_vector, 2)
        dot_place_city = tf.tensordot(place_vector, city_vector, 2)
        dot_place_category = tf.tensordot(place_vector, category_vector, 2)
        dot_user_price = tf.tensordot(user_vector, price_vector, 2)
        dot_place_price = tf.tensordot(place_vector, price_vector, 2)

        x = (dot_user_place + user_bias + place_bias +
             dot_user_city + dot_user_category +
             dot_place_city + dot_place_category +
             dot_user_price + dot_place_price)
        return tf.nn.sigmoid(x)

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=10,
    restore_best_weights=True
)

# Compile and train the model
model = RecommenderNet(num_users, num_place, num_cities, num_categories, 100)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(x=x_train, y=y_train, batch_size=8, epochs=100, validation_data=(x_val, y_val), callbacks=[early_stopping])

# Plot training history
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Metrics')
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Generate recommendations
user_id = df['User_Id'].sample(1).iloc[0]
place_visited_by_user = df[df['User_Id'] == user_id]
place_not_visited = tourism_new[~tourism_new['id'].isin(place_visited_by_user['Place_Id'].values)][['id', 'city_encoded', 'category_encoded', 'price', 'clipped_price', 'normalized_price']]
place_not_visited = place_not_visited.drop_duplicates()

place_not_visited_encoded = [[place_to_place_encoded.get(p_id), city, category, price, clipped_price, normalized_price] for p_id, city, category, price, clipped_price, normalized_price in place_not_visited.values]
user_encoder = user_to_user_encoded.get(user_id)
user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited_encoded), place_not_visited_encoded))

ratings = model.predict(user_place_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_place_ids = [place_encoded_to_place.get(place_not_visited_encoded[x][0]) for x in top_ratings_indices]

print('Showing recommendations for user: {}'.format(user_id))
print('===' * 9)
print('Places with high ratings from user')
print('----' * 8)
top_place_user = place_visited_by_user.sort_values(by='Place_Ratings', ascending=False).head(5)['Place_Id'].values
place_df_rows = tourism_new[tourism_new['id'].isin(top_place_user)]
print(place_df_rows[['name', 'category', 'city', 'description', 'price', 'lat', 'lng']])

print('----' * 8)
print('Top 10 place recommendations')
print('----' * 8)
recommended_place = tourism_new[tourism_new['id'].isin(recommended_place_ids)]
print(recommended_place[['name', 'category', 'city', 'description', 'price', 'lat', 'lng']])

# Save the model
# model.save(r"place_recommendation2.h5")