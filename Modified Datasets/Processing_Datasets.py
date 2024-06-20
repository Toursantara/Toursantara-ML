import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
info_tourism = pd.read_csv(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\ML_Repo\Datasets\tourism_with_id.csv")
tourism_rating = pd.read_csv(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\ML_Repo\Datasets\tourism_rating.csv")

# Identify outliers and clip the "Price" column
Q1 = info_tourism['Price'].quantile(0.25)
Q3 = info_tourism['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Clip outliers
info_tourism['Clipped_Price'] = info_tourism['Price'].clip(lower=lower_bound, upper=upper_bound)

# Normalize the clipped Price column
min_price = min(info_tourism['Clipped_Price'])
max_price = max(info_tourism['Clipped_Price'])
info_tourism['Normalized_Price'] = (info_tourism['Clipped_Price'] - min_price) / (max_price - min_price)

# Encode the City and Category columns
city_encoder = LabelEncoder()
info_tourism['City_Encoded'] = city_encoder.fit_transform(info_tourism['City'])
category_encoder = LabelEncoder()
info_tourism['Category_Encoded'] = category_encoder.fit_transform(info_tourism['Category'])

city_to_city_encoded = {x: i for i, x in enumerate(city_encoder.classes_)}
city_encoded_to_city = {i: x for i, x in enumerate(city_encoder.classes_)}
category_to_category_encoded = {x: i for i, x in enumerate(category_encoder.classes_)}
category_encoded_to_category = {i: x for i, x in enumerate(category_encoder.classes_)}

# Merge city_encoded, category_encoded, and Price into tourism_rating
tourism_rating = pd.merge(tourism_rating, info_tourism[['Place_Id', 'City_Encoded', 'Category_Encoded', 'Clipped_Price', 'Normalized_Price']], on='Place_Id', how='left')

# Create DataFrame for tourism
tourism_new = pd.DataFrame({
    "id": info_tourism.Place_Id.tolist(),
    "name": info_tourism.Place_Name.tolist(),
    "category": info_tourism.Category.tolist(),
    "description": info_tourism.Description.tolist(),
    "city": info_tourism.City.tolist(),
    "city_category": info_tourism[['City', 'Category']].agg(' '.join, axis=1).tolist(),
    "city_encoded": info_tourism['City_Encoded'].tolist(),
    "category_encoded": info_tourism['Category_Encoded'].tolist(),
    "lat": info_tourism['Lat'].tolist(),
    "lng": info_tourism['Long'].tolist(),
    "price": info_tourism['Price'].tolist(),
    "clipped_price": info_tourism['Clipped_Price'].tolist(),
    "normalized_price": info_tourism['Normalized_Price'].tolist()
})

tourism_new.to_csv(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\ML_Repo\Modified Datasets\tourism_new.csv", index=False)
tourism_rating.to_csv(r"C:\Users\fabia\Documents\Tugas Bangkit\Capstone Deployment\ML_Repo\Modified Datasets\tourism_rating_modified.csv", index=False)