import numpy as np
import pandas as pd
import os
from cleanco import basename
import re
import Levenshtein
import time
from collections import Counter


DATA_FOLDER         = os.path.join('.', 'data')
FACEBOOK_DATA_SET   = os.path.join(DATA_FOLDER, 'facebook_dataset.csv')
GOOGLE_DATA_SET     = os.path.join(DATA_FOLDER, 'google_dataset.csv')
WEBSITE_DATA_SET    = os.path.join(DATA_FOLDER, 'website_dataset.csv')


def strip_and_lower_values(df):
    '''
        Strips and lowers string values in a dataframe.

        Parameters:
        -----------
            df: the dataframe the actions will be performed on.

        Returns:
        --------
            The dataframe with lowered and striped values.
    '''
    return df.map(lambda x: x.lower().strip() if isinstance(x, str) else x) 

def read_facebook_dataset(path, verbose=True):
    '''
        Reads the facebook dataset from the specified path.\n
        If verbose is set True it prints a sample and the df.describe() function.

        Parameteres:
        ------------
            path: Path to the facebook dataset csv file.

        Returns:
        --------
            A Dataframe from the facebook dataset.
    '''
    dtype_dict = {
        'phone': str
    }

    df_facebook = pd.read_csv(path, escapechar='\\', dtype=dtype_dict)

    if verbose:
        print(">> FACEBOOK DATASET")
        print("Length of the facebook dataset: ", len(df_facebook))
        print("Sample from the facebook dataset:")
        print(df_facebook.sample(5))

        print("\nDescription of the facebook dataset:")
        print(df_facebook.describe())
        print()

    return df_facebook

def read_google_dataset(path, verbose=True):
    '''
        Reads the google dataset from the specified path.\n
        If verbose is set True it prints a sample and the df.describe() function.

        Parameteres:
        ------------
            path: Path to the google dataset csv file. 
        Returns:
        --------
            A Dataframe from the website dataset.
    '''
    dtype_dict = {
        'phone': str
    }

    df_google = pd.read_csv(path, escapechar='\\', dtype=dtype_dict)

    if verbose:
        print(">> GOOGLE DATASET")
        print("Length of the google dataset: ", len(df_google))
        print("\nSample from the google dataset:")
        print(df_google.sample(5))

        print("\nDescription of the google dataset:")
        print(df_google.describe())


    return df_google

def read_website_dataset(path, verbose=True):
    '''
        Reads the google dataset from the specified path.\n
        If verbose is set True it prints a sample and the df.describe() function.

        Parameteres:
        ------------
            path: Path to the website dataset csv file. 
        Returns:
        --------
            A Dataframe from the website dataset.
    '''
    df_website = pd.read_csv(path, sep=';')

    if verbose:
        print(">> WEBSITE DATASET")
        print("Length of the website dataset: ", len(df_website))
        print("\nSample from the website dataset:")
        print(df_website.sample(10))

        print("\nDescription of the website dataset:")
        print(df_website.describe())

    return df_website

def prepare_facebook_dataset(df_facebook):
    '''
        Prepare Facebook data by renaming columns, splitting categories, and reordering columns.
    '''
    final_column_order = [
        'domain', 'company_name_facebook', 'address_facebook', 'categories_facebook', 'category_list_facebook', 
        'city_facebook', 'country_code_facebook', 'country_name_facebook', 'phone_facebook', 'phone_country_code_facebook',
        'region_code_facebook', 'region_name_facebook', 'zip_code_facebook', 'email', 'link_facebook', 'page_type_facebook', 'description_facebook'
    ]
    
    # split the category column into list of categories
    df_facebook['category_list'] = df_facebook['categories'].str.split('[&,|-]| and ')
    df_facebook['category_list'] = df_facebook['category_list'].apply(
        lambda categories: np.sort(np.unique([category.lower().strip() for category in categories])) if str(categories) != 'nan' else categories
    )

    # rename columns
    suffix = '_facebook'
    df_facebook.columns = [col + suffix for col in df_facebook.columns]
    df_facebook.rename(columns={
        'domain_facebook':  'domain',
        'name_facebook': 'company_name_facebook',
        'email_facebook': 'email'
    }, inplace=True)
    # reorder columns
    df_facebook = df_facebook[final_column_order]
    
    df_facebook.sample(1000).to_csv('sample_facebook.csv')
    
    return df_facebook


def prepare_google_dataset(df_google):
    '''
        Prepare Google data by renaming coulmns, cleansing addresses, splitting categories, and reordering columns.
    '''
    final_column_order = [
        'domain', 'company_name_google', 'address_google', 'categories_google', 'category_list_google', 
        'city_google', 'country_code_google', 'country_name_google', 'phone_google', 'phone_country_code_google', 
        'region_code_google', 'region_name_google', 'zip_code_google', 'raw_address_google', 'raw_phone_google', 'description_google', 'trust_only_google'
    ]
    # Remove the unnecessary part from the address and raw address columns
    regex = r'(\d+)\+ years in business · '
    df_google['address'] = df_google['address'].str.replace(regex, '', regex=True)
    df_google['raw_address'] = df_google['raw_address'].str.replace(regex, '', regex=True)

    # Remove quotes from the address columns
    regex = r'\".*\"'
    df_google['address'] = df_google['address'].str.replace(regex, '', regex=True)
    df_google['raw_address'] = df_google['raw_address'].str.replace(regex, '', regex=True)

    # split the category column into list of categories
    # The categories are separated by a '& , and -' convert them into a list
    df_google['category_list'] = df_google['category'].str.split('[&,-]| and ')
    # strip the values, make them lowercase and sort them
    df_google['category_list'] = df_google['category_list'].apply(
        lambda categories: np.sort([category.lower().strip() for category in categories]) if str(categories) != 'nan' else categories
    )

    # Count the presence of each domain
    domain_counts = df_google['domain'].value_counts()
    # Select domains that appear more than 500 times
    threshold = 500
    large_domains = domain_counts[domain_counts >= threshold].index.tolist()
    
    df_google['trust_only'] = (df_google['domain'].isin(large_domains))

    # rename columns
    suffix = '_google'
    df_google.columns = [col + suffix for col in df_google.columns]

    df_google.rename(columns={
        'domain_google': 'domain',
        'name_google': 'company_name_google',
        'category_google': 'categories_google',
        'text_google': 'description_google'
    }, inplace=True)
    # reorder columns
    df_google = df_google[final_column_order]
    
    df_google.sample(1000).to_csv('sample_google.csv')

    return df_google    

def prepare_website_dataset(df_website):
    '''
        Prepare Website data by renaming columns, selecting company name, splitting categories, and reordering columns.
    '''
    final_column_order = [
        'domain', 'company_name_website', 'legal_name_website', 'site_name_website', 'categories_website', 'category_list_website',
        'city_website', 'country_name_website', 'region_name_website', 'phone_website', 'domain_suffix_website', 'language_website', 'tld_website'
    ]

    # function to select the proper company name
    def choose_proper_name(row):
        if pd.isnull(row['legal_name']):
            return row['site_name']
        elif pd.isnull(row['site_name']):
            return row['legal_name']
        else:
            # this might be improved, but in the most cases this is more accurate than 'legal_name'
            return row['site_name']

    # Apply the function to create the 'company_name' column
    df_website['company_name'] = df_website.apply(choose_proper_name, axis=1)

    df_website['category_list'] = df_website['s_category'].str.split('[&,-]| and ')
    df_website['category_list'] = df_website['category_list'].apply(
        lambda categories:  sorted([category.lower().strip() for category in categories]) if str(categories) != 'nan' else categories
    )

    # rename columns
    suffix = '_website'
    df_website.columns = [col + suffix for col in df_website.columns]

    df_website.rename(columns={
        'root_domain_website': 'domain',
        's_category_website': 'categories_website', 
        'main_city_website': 'city_website',
        'main_country_website': 'country_name_website',
        'main_region_website': 'region_name_website',
    }, inplace = True)
    # reorder columns 
    df_website = df_website[final_column_order]
    df_website.sample(1000).to_csv('sample_website.csv')
    
    return df_website

def count_column_values_in_dataset(df_dataset, column_name, dataset_name):
    '''
        Counts the presence of column values in a dataset.

        Parameters:
        -----------
            df_dataset: Dataframe in which the values will be counted.
            column_name: The name of the column in the dataframe in which the values wanted to be counted.
            dataset_name: The name of the dataset to appear in the result.

        Returns:
        --------
            A dataframe which holds the counts of every unique item in the original dataframe.
    '''
    count_dataset = Counter(df_dataset[column_name])

    columns=[column_name, f'nr_in_{dataset_name}']
    rows = []
    for key, value in count_dataset.items():
        row = {
            columns[0]: key,
            columns[1]: value
        }
        rows.append(row)
    
    return pd.DataFrame(rows, columns=columns)

def column_values_counts_in_datasets(df_google, df_facebook, df_website, column_name):
    '''
        Calculates the unique values and counts of a specified column in three datasets (Google, Facebook, and Website).

        Parameters:
        -----------
            df_google: DataFrame containing Google dataset.
            df_facebook: DataFrame containing Facebook dataset.
            df_website: DataFrame containing Website dataset.
            column_name: Name of the column for which unique values and counts will be calculated.

        Returns:
        --------
            A DataFrame with the unique values and counts of the specified column
            in each dataset, merged into a single DataFrame.
    '''
    google_domains   = set(df_google[column_name].unique())
    facebook_domains = set(df_facebook[column_name].unique())
    website_domains  = set(df_website[column_name].unique())

    google_differences   = google_domains.difference(facebook_domains, website_domains)
    facebook_differences = facebook_domains.difference(google_domains, website_domains)
    website_differences  = website_domains.difference(google_domains, facebook_domains)

    # Print the differences or do further processing
    print("Unique to google:", google_differences)
    print("Unique to facebook:", facebook_differences)
    print("Unique to website:", website_differences)

    domains_in_google = count_column_values_in_dataset(df_google, 'domain', 'google')
    domains_in_facebook = count_column_values_in_dataset(df_facebook, 'domain', 'facebook')
    domains_in_website = count_column_values_in_dataset(df_website, 'domain', 'website')

    merged_data = pd.merge(domains_in_google, domains_in_facebook, on=column_name, how='outer')
    merged_data = pd.merge(merged_data, domains_in_website, on=column_name, how='outer')

    return merged_data


def merge_datasets_by_domains(df_google, df_facebook, df_website):
    '''
        Unites the three dataset by the domain columns and creates a final DataFrame from it.
    '''
        
    google_domains   = set(df_google['domain'].unique())
    facebook_domains = set(df_facebook['domain'].unique())
    website_domains  = set(df_website['domain'].unique())

    domains = set.union(google_domains, facebook_domains, website_domains)

    final_columns = ['domain', 'company_name', 'phone', 'email', 'zip_code', 'category_list', 'address', 'city', 'country_code', 'country_name', 'region_code', 'region_name']

    # domain = 'mbwd.ca'
    # google_subset = df_google[df_google['domain'] == domain]
    # facebook_subset = df_facebook[df_facebook['domain'] == domain]
    # website_subset = df_website[df_website['domain'] == domain]

    # print(google_subset)
    # print(facebook_subset)
    # print(website_subset)

    # google_subset.to_csv('subset_google.csv')
    # facebook_subset.to_csv('subset_facebook.csv')
    # website_subset.to_csv('subset_website.csv')

    google_subset = df_google
    facebook_subset = df_facebook
    website_subset = df_website

    # MERGING  
        
    # merge the three dataset by the domain column
    merged_df = facebook_subset.merge(website_subset, on='domain', how='outer')
    merged_df = merged_df.merge(google_subset, on='domain', how='outer')

    # choose the google address if set, facebook address otherwise
    merged_df['address'] = merged_df['address_google'].combine_first(merged_df['address_facebook'])


    # In some cases this is the good approach, however in others it is not the best
    def choose_between_multiple(row, column_names, trust_google_only):
        '''
            Chooses the most accurate value in a row from multiple columns
        '''

        # if there are less than three value choose the first non null
        non_null_values = row.dropna()
        

        if len(non_null_values) ==  1 or len(non_null_values) == 2:
            return non_null_values.iloc[0]
        if len(non_null_values) == 0:
            return row[column_names[0]]

        if trust_google_only == True :
            return row['company_name_google']

        # Check if the Google name has a Human name in it
        title_pattern =  r'(M\.?D\.?)|(Ph\.?D)|([A-Z]\.[A-Z]\.)|(\b[A-Z]{2}\b)'
        if len(re.findall(title_pattern, row['company_name_google'])) > 0:
            if str(row['company_name_facebook']) == 'nan' or row['company_name_facebook'] == '':
                return row['company_name_website']
            else:  
                return row['company_name_facebook']
        
        # 
        distances = {name: 0 for name in column_names}

        for i in range(len(column_names)):
            if str(row[column_names[i]]) == 'nan':
                row[column_names[i]] = ""
            for j in range(i + 1, len(column_names)):
                if str(row[column_names[j]]) == 'nan':
                    row[column_names[j]] = ""

                distances[column_names[i]] += Levenshtein.distance(row[column_names[i]], row[column_names[j]])
                distances[column_names[j]] += Levenshtein.distance(row[column_names[j]], row[column_names[i]])

        return row[min(comps, key=lambda comp: distances[comp])]
    
    comps = ['company_name_google', 'company_name_facebook', 'company_name_website']
    merged_df['company_name'] = merged_df[['company_name_google', 'company_name_facebook', 'company_name_website', 'trust_only_google']].apply(
        lambda row: choose_between_multiple(row[comps], comps, row['trust_only_google']), 
        axis=1
    )
    
    # combine the categories from the three dataset
    def combine_categories(row):
        category_sets = [set(cat) for cat in row if str(cat) != 'nan']
        combined_categories = set.union(*category_sets) if category_sets else set()
        # TODO: further refinement might be needed based on semantic compatibility 
        return list(combined_categories)
    merged_df['category_list'] = merged_df[['category_list_google', 'category_list_facebook', 'category_list_website']].apply(
        combine_categories,
        axis=1
    )

    # 
    merged_df['city'] =  merged_df['city_google'].combine_first(merged_df['city_facebook']).combine_first(merged_df['city_website'])
    merged_df['country_code'] = merged_df['country_code_google'].combine_first(merged_df['country_code_facebook'])
    merged_df['country_name'] = merged_df['country_name_google'].combine_first(merged_df['country_name_facebook']).combine_first(merged_df['country_name_website'])
    merged_df['region_code'] = merged_df['region_code_google'].combine_first(merged_df['region_code_facebook'])
    merged_df['region_name'] = merged_df['region_name_google'].combine_first(merged_df['region_name_google']).combine_first(merged_df['region_name_website'])
    merged_df['zip_code'] = merged_df['zip_code_google'].combine_first(merged_df['zip_code_facebook'])
    merged_df['phone'] = merged_df['phone_google'].combine_first(merged_df['phone_facebook']).combine_first(merged_df['phone_website'])
    merged_df['phone_country_code'] = merged_df['phone_country_code_google'].combine_first(merged_df['phone_country_code_facebook'])
    #


    # sort columns so we can see the differences between the datasets values
    columns = np.sort(merged_df.columns)
    merged_df = merged_df[columns]

    merged_df.drop_duplicates(subset=['address', 'phone'],  keep='first', inplace=True)
    
    merged_df.sort_values(by=['domain'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    merged_df.to_csv('merged.csv', index=False)   
    merged_df[final_columns].to_csv('merged_final.csv', index=False)     
    

if __name__ == "__main__":
    pd.set_option('display.max_colwidth', 25)
    verbose = True
    
    df_facebook = read_facebook_dataset(FACEBOOK_DATA_SET, verbose)
    df_google   = read_google_dataset(GOOGLE_DATA_SET, verbose)
    df_website  = read_website_dataset(WEBSITE_DATA_SET, verbose)

    df_google   = prepare_google_dataset(df_google)
    df_facebook = prepare_facebook_dataset(df_facebook)
    df_website  = prepare_website_dataset(df_website)

    counts_file_name = 'domain_counts.csv'
    if not os.path.exists(counts_file_name):
        df_domains = column_values_counts_in_datasets(df_google, df_facebook, df_website, 'domain')
        df_domains.to_csv(counts_file_name)

    merge_datasets_by_domains(df_google, df_facebook, df_website)
    
    df_merged = pd.read_csv('merged.csv')
    df_merged.sample(min(len(df_merged), 10000)).to_csv('merged_sample.csv')

    df_final = pd.read_csv('merged_final.csv', dtype={'phone': str})
    print(df_final.describe())
    

    



    

