

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Read the Dataset of the Population Growth
df1 = pd.read_csv("/content/API_SP.POP.GROW_DS2_en_csv_v2_5358698.csv")
df1.head()

# Read the Dataset of the Unemployement rate 
df2 = pd.read_csv("/content/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_5358416.csv")
df2.head()

def convert_to_years(df1):
    id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    value_vars = df1.columns.difference(id_vars).tolist()
    df1_years = pd.melt(df1, id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Value')
    df1_years['Year'] = pd.to_datetime(df1_years['Year'], format='%Y')
    return df1_years['Year']

def convert_to_dataframe(df1):
    years = convert_to_years(df1)
    return df1['Country Name'], years

Country_Name, Year = convert_2_datafram(df1)

print(Country_Name)
print(Year)

"""To begin exploring the Population and Unemployment dataset, first check its available columns and then generate a summary using the .describe method."""

df1.columns

# Calculate summary statistics
print(df1.describe())

# Calculate summary statistics
print(df2.describe())

# Assuming the dataset is stored in a pandas DataFrame called 'df1'
# Extract the year columns for which you want to calculate the median
year_columns = ['1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
                '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
                '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
                '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
                '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
                '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

# Extract the data for the year columns
year_data = df1[year_columns]

# Calculate the correlation matrix
correlation_matrix = year_data.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Assuming the dataset is stored in a pandas DataFrame called 'df1'
# Extract the year columns for which you want to calculate the median
year_columns = ['1961', '1965', '1970', '1975', '1980', '1985','1990', '1995', '2000', 
                '2005', '2010', '2015', '2020']

# Calculate the median for each year column
median_values = df1[year_columns].median()

# Create a line graph
plt.plot(year_columns, median_values)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Median Population Growth')
plt.title('Median Population Growth over the Years')

# Display the graph
plt.show()

"""To compare indicators across different countries over time and explore their interdependence, create a bar chart, line chart, and correlation matrix. To gain deeper insights not only among countries but also among indicators, use a Choropleth Ma and scatter Plot"""

# # Load the dataset
# df = pd.read_csv('/content/API_SP.POP.GROW_DS2_en_csv_v2_5358698.csv')

# create a list of countries to select
countries_to_select = ['USA', 'RUS', 'IND', 'PAK', 'AUS', 'BEL','CHN', 'IDN','CAN', 'FRA','ARB','BRA' ]

# filter the dataset by the selected countries
selected_df1 = df1[df1['Country Code'].isin(countries_to_select)]

# Select the columns of interest
selected_df1 = selected_df1[['Country Code', '1991']]

# Sort the data in descending order
selected_df1 = selected_df1.sort_values(by='1991', ascending=False)

# Create the bar chart
plt.figure(figsize=(8,6))
plt.bar(selected_df1['Country Code'], selected_df1['1991'])
plt.xlabel('Country')
plt.ylabel('Population Growth (Anual %)')
plt.title('Population Growth in 1991')
plt.show()

df1.columns

# filter the dataset by the selected countries
selected_df1 = df1[df1['Country Code'].isin(countries_to_select)]
# Select the columns of interest
selected_df1 = selected_df1[['Country Code', '2019']]

# Sort the data in descending order
selected_df1 = selected_df1.sort_values(by='2019', ascending=False)

# Create the bar chart
plt.figure(figsize=(8,6))
plt.bar(selected_df1['Country Code'], selected_df1['2019'])
plt.xlabel('Country')
plt.ylabel('Population Growth (Anual %)')
plt.title('Population Growth in 2019')
plt.show()

# create a list of countries to select
countries_to_select2 = ['USA', 'RUS', 'IND', 'PAK', 'AUS', 'BEL','CHN', 'IDN','CAN', 'FRA','ARB','BRA' ]

# filter the dataset by the selected countries
selected_df2 = df2[2['Country Code'].isin(countries_to_select2)]

# Select the columns of interest
selected_df2 = selected_df2[['Country Code', '1991']]

# Sort the data in descending order
selected_df2 = selected_df2.sort_values(by='1991', ascending=False)

# Create the bar chart
plt.figure(figsize=(8,6))
plt.bar(selected_df2['Country Code'], selected_df2['1991'])
plt.xlabel('Country')
plt.ylabel('Unemployment (Anual %)')
plt.title('Unemployment in 1991, total (% of total labor force) (modeled ILO estimate)')
plt.show()

# filter the dataset by the selected countries
selected_df2 = df2[df2['Country Code'].isin(countries_to_select)]

# Select the columns of interest
selected_df2 = selected_df2[['Country Code', '2020']]

# Sort the data in descending order
selected_df2 = selected_df2.sort_values(by='2020', ascending=False)

# Create the bar chart
plt.figure(figsize=(8,6))
plt.bar(selected_df2['Country Code'], selected_df2['2020'])
plt.xlabel('Country')
plt.ylabel('Unemployment (Anual %)')
plt.title('Unemployment in 2020, total (% of total labor force) (modeled ILO estimate)')
plt.show()

# Select countries of interest
countries = ['USA', 'RUS', 'IND', 'PAK', 'AUS', 'BEL','CHN', 'IDN','CAN', 'FRA','ARB','BRA' ]

# Subset the data for these countries
subset = df1[df1["Country Code"].isin(countries)]

# Set the index to be the country names
subset.set_index("Country Code", inplace=True)

# Select columns of interest
cols = [ '1990', '1995', '2000','2005', '2010', '2015', '2020']
subset = subset[cols]

# Plot the data
subset.T.plot(kind='line', figsize=(10,6))
plt.title('Population Growth')
plt.xlabel('Year')
plt.ylabel('Population Growth (% of Annual)')
plt.show()

# Subset the data for these countries
subset1 = df2[df2["Country Code"].isin(countries)]

# Set the index to be the country names
subset1.set_index("Country Code", inplace=True)

# Select columns of interest
cols = [ '1991', '1995', '2000','2005', '2010', '2015', '2020']
subset1 = subset1[cols]

# Plot the data
subset1.T.plot(kind='line', figsize=(10,6))
plt.title('Unemployment, total (% of total labor force) (modeled ILO estimate) ')
plt.xlabel('Year')
plt.ylabel('Unemployment, total (% of total labor force) (modeled ILO estimate)')
plt.show()

"""**Explore and understand any correlations (or lack of) between indicators. Does this vary between country, have any correlations or trends changed with time?**"""

# Calculate correlation matrix
corr_matrix = df1.corr()

# Plot correlation matrix using heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Create Choropleth Map
fig = px.choropleth(df1, locations='Country Code', color='1980',
                    hover_name='Country Name',
                    projection='mercator',
                    title='World Population Growth in 1980')
fig.show()

# Create Choropleth Map
fig = px.choropleth(df1, locations='Country Code', color='2020',
                    hover_name='Country Name',
                    projection='mercator',
                    title='World Population Growth in 2020')
fig.show()

# Extract the columns for population and unemployment rate
population_data = df1['1991']
unemployment_rate_data = df2['1991']

# Plot the scatter plot
plt.scatter(population_data, unemployment_rate_data)

# Add labels and title
plt.xlabel('Population')
plt.ylabel('Unemployment Rate')
plt.title('Scatter Plot: Population vs Unemployment Rate ')

# Show the plot
plt.show()

# Extract the columns for population and unemployment rate
population_data = df1['2020']
unemployment_rate_data = df2['2020']

# Plot the scatter plot
plt.scatter(population_data, unemployment_rate_data)

# Add labels and title
plt.xlabel('Population')
plt.ylabel('Unemployment Rate')
plt.title('Scatter Plot: Population vs Unemployment Rate')

# Show the plot
plt.show()

