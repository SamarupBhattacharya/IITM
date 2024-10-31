The following are the 3 bullet points which describe my method of processing the dataset:
* I used the GitHub API to fetch user data and their repositories. Parsed the JSON response, processed it with Python, then saved the results in a CSV format using Pandas. Cleaned up missing values and converted booleans to lower case.
* The most interesting fact I discovered after analyzing the data is that on average Fluent repositories have the highest number of stars compared to all other languages. This is very surprising considering that Fluent has very limited use cases.
* Analysis of the data reveals that on and average, hireable people were following 46.136 more people as compared to non-hireable people. Thus, this creates a clear incentive for software engineers to collaborate with more number of people to enhance their skills and thereby make them more hireable.

I am including all the snippets of code that I used for scraping as well as analysis of the data for your kind perusal.
**Scraping of data obtained from https://api.gitgub.com/{username}**
import requests\n
import pandas as pd
import json
**#GitHub API token**
GITHUB_TOKEN = '********************************************************************************'

# GitHub API endpoint for searching users
url = 'https://api.github.com/search/users'

# Parameters for the search query
params = {
    'q': 'location:Berlin followers:>=200',
    'per_page': 100,
    'access_token': GITHUB_TOKEN
}
# Adding the Authorization
headers = { 'Authorization': f'token {GITHUB_TOKEN}' }

# List to store user data
users = []

# Pagination to handle more than 100 results
page = 1
while True:
    response = requests.get(url, params={**params, 'page': page}, headers=headers)
    print(response);
    response_json = response.json()
    for user in response_json['items']:
        users.append(
            user['login']
        )
    if 'next' in response.links:
        page += 1
        params['page'] = page
    else:
        break
print(users)
# Function to get user details
def get_user_details(username):
     url = f'https://api.github.com/users/{username}'
     response = requests.get(url, headers=headers)
     user_data = response.json()
     # Process company name
     company = user_data.get('company', 'N/A')
     if company!='N/A' and company!=None:
        company = company.strip().lstrip('@').upper()
     return { "login": user_data.get('login', 'N/A'), "name": user_data.get('name', 'N/A'), "company": company, "location": user_data.get('location', 'N/A'), "email": user_data.get('email', 'N/A'), "hireable": user_data.get('hireable', 'N/A'), "bio": user_data.get('bio', 'N/A'), "public_repos": user_data.get('public_repos', 'N/A'), "followers": user_data.get('followers', 'N/A'), "following": user_data.get('following', 'N/A'), "created_at": user_data.get('created_at', 'N/A') }
# Fetching details for each user
detailed_users = [get_user_details(user) for user in users]
print(1)
import csv
 # now we will open a file for writing
data_file = open('users.csv', 'w')

# create the csv writer object
csv_writer = csv.writer(data_file)

# Counter variable used for writing headers to the CSV file
count = 0

for emp in detailed_users:
    if count == 0:
         # Writing headers of CSV file
        header = list(emp.keys())
        print(header)
        csv_writer.writerow(header)
        count += 1
    # Writing data of CSV file
    csv_writer.writerow(emp.values())
    print(emp)

data_file.close()
print(2)
import pandas as pd

# Load CSV file
df = pd.read_csv('users (2).csv')

def convert_hireable(value):
  if pd.isna(value) or value == '':
    return 'false'
  elif value is True:
    return 'true'
  return value
df['hireable'] = df['hireable'].apply(convert_hireable)
# Save the updated DataFrame back to the CSV file
df.to_csv('users (2).csv', index=False)
**Scraping the data obtained from https://api.github.com/{username}/repos**
import requests
import pandas as pd

# GitHub API URL
GITHUB_API_URL = "https://api.github.com/users/{}/repos"

# Read the users from the CSV file
users_df = pd.read_csv("users.csv")

# Initialize an empty list to store repository data
repos_data = []

# Loop through each user and fetch their repositories
for index, row in users_df.iterrows():
    user_login = row['login']
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(GITHUB_API_URL.format(user_login), headers=headers)
    repos = response.json()
    repos = repos[:500] if len(repos) > 500 else repos
    for repo in repos:
        repos_data.append({
            'login': repo['owner']['login'],
            'full_name': repo['full_name'],
            'created_at': repo['created_at'],
            'stargazers_count': repo['stargazers_count'],
            'watchers_count': repo['watchers_count'],
            'language': repo['language'],
            'has_projects': repo['has_projects'],
            'has_wiki': repo['has_wiki'],
            'license_name': repo['license']['name'] if repo['license'] is not None and repo['license']['name'] else ''
        })

# Create a DataFrame from the repository data
repos_df = pd.DataFrame(repos_data)

# Save the DataFrame to a CSV file
repos_df.to_csv("repositories.csv", index=False)
**Code for Q1**
sorted_df = df.sort_values(by='followers', ascending=False)
top_5 = sorted_df.head(5)
print(','.join(top_5['login'].tolist()))
**Code for Q2**
import pandas as pd
df = pd.read_csv('users.csv')
sorted_df = df.sort_values(by='created_at', ascending=True)
top_5 = sorted_df.head(5)
print(','.join(top_5['login'].tolist()))
**Code for Q3**
import pandas as pd
df = pd.read_csv("repositories.csv")
filtered_df = df[df['license_name'] != '']
license_counts = filtered_df['license_name'].value_counts()
top_3_licenses = license_counts.head(3).index.tolist()
top_3_licenses_str = ','.join(top_3_licenses)
print(top_3_licenses_str)
**Code for  Q4**
most_frequent_company = df['company'].mode()[0]
print(f'The most frequent company is: {most_frequent_company}')
**Code for Q5**
import pandas as pd
df = pd.read_csv("repositories.csv")
filtered_df = df[df['language'] != '']
language_counts = filtered_df['language'].value_counts()
most_popular_language = language_counts.idxmax()
print(f'The most popular programming language is: {most_popular_language}')
**Code for Q6**
import pandas as pd
users_df = pd.read_csv('users (2).csv')
repos_df = pd.read_csv('repositories.csv')
users_df['created_at'] = pd.to_datetime(users_df['created_at'])
users_after_2020 = users_df[users_df['created_at'] > '2020-01-01']
logins_after_2020 = users_after_2020['login']
filtered_repos = repos_df[repos_df['login'].isin(logins_after_2020)]
language_counts = filtered_repos['language'].value_counts()
second_most_popular_language = language_counts.index[1]
print(f'The second most popular programming language is: {second_most_popular_language}')
**Code for Q7**
import pandas as pd
repos_df = pd.read_csv('repositories.csv')
filtered_repos = repos_df[repos_df['language'] != '']
average_stars = filtered_repos.groupby('language')['stargazers_count'].mean()
most_popular_language = average_stars.idxmax()
highest_average_stars = average_stars.max()
print(f'The language with the highest average number of stars per repository is: {most_popular_language} with an average of {highest_average_stars:.2f} stars.')
**Code for 8**
df = pd.read_csv('users.csv')
print(df.loc[df['login'] == 'tiangolo', 'hireable'].values[0]=="True")
df['leader_strength'] = df['followers'] / (1 + df['following'])
sorted_df = df.sort_values(by='leader_strength', ascending=False)
top_5 = sorted_df.head(5)
top_5_logins = ','.join(top_5['login'].tolist())
print(top_5_logins)
**Code for Q9**
correlation = df['followers'].corr(df['public_repos'])
print(f'The correlation between the number of followers and the number of public repositories is: {correlation:.3f}')
**Code for Q10**
from sklearn.linear_model import LinearRegression
X = df[['public_repos']]
y = df['followers']
model = LinearRegression()
model.fit(X, y)
coef = model.coef_[0]
print(f'Estimated additional followers per additional public repository: {coef:.3f}')
**Code for Q11**
import pandas as pd
repos_df = pd.read_csv('repositories.csv')
correlation = repos_df['has_projects'].astype(int).corr(repos_df['has_wiki'].astype(int))
print(f'The correlation between having projects enabled and having a wiki enabled is: {correlation:.3f}')
**Code for Q12**
avg_following_hireable = df[df['hireable'] == True]['following'].mean()
avg_following_non_hireable = df[df['hireable'].isnull()]['following'].mean()
print(f'DIfference of average followers: {(avg_following_hireable-avg_following_non_hireable):.3f}')
**Code for Q13**
from sklearn.linear_model import LinearRegression
import numpy as np
df['bio_word_count'] = df['bio'].apply(lambda x: len(str(x).split()))
X = df['bio_word_count'].values.reshape(-1, 1)
y = df['followers'].values
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
print(f'Regression slope: {slope:.3f}')
**Code for Q14**
import pandas as pd
repos_df = pd.read_csv('repositories.csv')
repos_df['created_at'] = pd.to_datetime(repos_df['created_at'], utc=True)
repos_df['day_of_week'] = repos_df['created_at'].dt.dayofweek
weekend_repos = repos_df[repos_df['day_of_week'].isin([5, 6])]
weekend_repos_count = weekend_repos['login'].value_counts()
top_5_weekend_creators = weekend_repos_count.head(5).index.tolist()
top_5_weekend_creators_str = ','.join(top_5_weekend_creators)
print(top_5_weekend_creators_str)

Thanks

