#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install sodapy')


# In[ ]:


import pandas as pd
from sodapy import Socrata


# In[ ]:


data_url = 'data.cityofnewyork.us'
app_token = 'DiYM4Qi0LWQfnpvC11ksbm5ju'

client = Socrata(data_url, app_token)
client.timeout = 240


# In[ ]:


for x in range(2021, 2025):
    # get data
    start = 0
    chunk_size = 2000
    results = []

    where_clause = f"complaint_type LIKE 'Taxi Complaint%' AND date_extract_y(created_date)={x}"
    data_set = 'erm2-nwe9'
    record_count = client.get(data_set, where=where_clause, select='COUNT(*)')
    print(f'Taxi Complaint data set from {x}')

    while True:
        results.extend(client.get(data_set, where=where_clause, offset=start, limit=chunk_size))
        start += chunk_size
        if (start > int(record_count[0]['COUNT'])):
            break

    # export data to csv
    df = pd.DataFrame.from_records(results)
    df.to_csv("311_Taxi_Complaint.csv", index=False)


# In[ ]:


from google.colab import drive
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# Assuming 'results' contains your data
# Create a DataFrame
df = pd.DataFrame.from_records(results)

# Specify the path within your Google Drive
file_path = '/content/drive/My Drive/CIS9350/311_Taxi_Complaint.csv'

# Save the DataFrame to a CSV file in the specified path
df.to_csv(file_path, index=False)

print(f"File saved successfully at {file_path}")


# In[ ]:


import os
print(os.getcwd())


# In[ ]:


import os
print(os.path.isfile("311_Taxi_Complaint.csv"))


# In[ ]:


df.to_csv("311_Taxi_Complaint.csv", index=False)
print("File saved successfully.")


# In[ ]:


import pandas as pd

# Load the saved CSV file
df = pd.read_csv("311_Taxi_Complaint.csv")

# Display the first few rows of the data
print(df.head())


# In[ ]:


get_ipython().system('pip install ydata-profiling # Install the ydata-profiling package using pip')


# In[ ]:


import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset (replace the file path if necessary)
df = pd.read_csv("311_Taxi_Complaint.csv")

# Generate the data profiling report
profile = ProfileReport(df, title="Taxi Complaint Data Profiling Report", explorative=True)

# Save the profiling report to an HTML file
profile.to_file("taxi_complaint_data_profiling_report.html")

# Optionally, you can display the report in a Jupyter notebook (if using Jupyter)
profile.to_notebook_iframe()


# In[ ]:


for x in range(2021, 2025):
    # get data
    start = 0
    chunk_size = 2000
    results = []

    where_clause = f"date_extract_y(inspection_date)={x}"
    data_set = '43nn-pn8j'
    record_count = client.get(data_set, where=where_clause, select='COUNT(*)')
    print(f'Restaurant Inspection data set from {x}')

    while True:
        results.extend(client.get(data_set, where=where_clause, offset=start, limit=chunk_size))
        start += chunk_size
        if (start > int(record_count[0]['COUNT'])):
            break

    # export data to csv
    df = pd.DataFrame.from_records(results)
    df.to_csv("Restaurant_Inspection.csv", index=False)


# In[ ]:


from google.colab import drive
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# Assuming 'results' contains your data
# Create a DataFrame
df = pd.DataFrame.from_records(results)

# Specify the path within your Google Drive
file_path = '/content/drive/My Drive/CIS9350/Restaurant_Inspection.csv'

# Save the DataFrame to a CSV file in the specified path
df.to_csv(file_path, index=False)

print(f"File saved successfully at {file_path}")


# In[ ]:


import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset (replace the file path if necessary)
df = pd.read_csv("Restaurant_Inspection.csv")

# Generate the data profiling report
profile = ProfileReport(df, title="Restaurant Inspection Data Profiling Report", explorative=True)

# Save the profiling report to an HTML file
profile.to_file("Restaurant_inspection_data_profiling_report.html")

# Optionally, you can display the report in a Jupyter notebook (if using Jupyter)
profile.to_notebook_iframe()


# **CREATING FUNCTIONS**

# In[ ]:


pip install google-cloud-bigquery google-auth


# In[ ]:


from google.colab import drive
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging
from datetime import datetime


# In[ ]:


# Constants for GCP and file paths
gcp_project = 'my-project-0577-434418'
bq_dataset = '311_taxi_complaints'
csv_file_path = '/content/drive/My Drive/CIS9350/311_Taxi_Complaint.csv'
log_file_dir = '/content/drive/My Drive/CIS9350/logs/'


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')  # Mount Google Drive

# Now specify the path to your key
key_path = '/content/drive/My Drive/CIS9350/my-project-0577-434418-59d2bed1a85c.json'

from google.cloud import bigquery
from google.oauth2 import service_account

# Construct credentials from the key file
credentials = service_account.Credentials.from_service_account_file(key_path)

# Initialize BigQuery client with explicit credentials
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Example: List BigQuery datasets in your project
datasets = client.list_datasets()
for dataset in datasets:
    print(f"Dataset: {dataset.dataset_id}")


# In[ ]:


def load_csv_data_file(logging, file_name, df):
    """ #load and transform data return back to
    load_csv_data_file
    Accepts a file source path and a file name
    Loads the file into a data frame
    Exits the program on error
    Returns the dataframe
    """
    logging.info(f'Reading source data file: {file_name}')
    try:
        df = pd.read_csv(file_name, low_memory=False)
        df = df.rename(columns=str.lower)
        logging.info(f'Read {len(df)} records from source data file: {file_name}')
        return df
    except Exception as e:
        logging.error(f'Failed to read file: {file_name}. Error: {str(e)}')
        raise


# In[ ]:


def load_csv_data_file(logging, file_path, df):
    try:
        df = pd.read_csv(file_path)  # Correct way to load CSV data into a DataFrame
        logging.info(f"Loaded {len(df)} records from {file_path}.")
    except Exception as e:
        logging.error(f"Error loading CSV file {file_path}: {e}")
    return df


# In[ ]:


def transform_data(logging, df, columns): # Add columns as an argument
    """
    transform_data
    Accepts a data frame
    Performs any specific cleaning and transformation steps on the dataframe
    Returns the modified dataframe
    """
    logging.info('Transforming dataframe.')
    df = df[columns] # Select columns
    df = df.drop_duplicates()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


# In[ ]:


def transform_data(logging, columns, df):
    """
    transform_data
    Accepts a data frame
    Performs any specific cleaning and transformation steps on the dataframe
    Returns the modified dataframe
    """
    logging.info('Transforming dataframe.')

    # Ensure `columns` is a list of column names, and filter the dataframe
    if isinstance(columns, list):
        df = df[columns]  # Access columns in DataFrame by name
    else:
        logging.warning("Expected 'columns' to be a list of column names.")
        return df

    # Remove duplicates if any
    df = df.drop_duplicates()

    # Ensure df is a DataFrame (if the input is not already a DataFrame)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    return df


# In[ ]:


def transform_data(logging, columns, df):
    # Ensure all required columns exist
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns in the DataFrame: {missing_columns}")
        raise ValueError(f"Missing columns in the DataFrame: {missing_columns}")


# In[ ]:


from google.cloud import bigquery
from google.oauth2 import service_account
import logging
import os

def create_bigquery_client(logging):
    """
    Creates a BigQuery client using the service account key from Google Drive.
    Returns the BigQuery client object or None if creation fails.
    """
    # Use environment variable or hardcoded path for the key file
    key_path = os.getenv('BIGQUERY_KEY_PATH', '/content/drive/My Drive/CIS9350/my-project-0577-434418-59d2bed1a85c.json')

    try:
        # Construct credentials from the key file
        credentials = service_account.Credentials.from_service_account_file(key_path)

        # Initialize BigQuery client with credentials
        bqclient = bigquery.Client(credentials=credentials, project=credentials.project_id)

        # Log client creation
        logging.info('Created BigQuery client: %s', bqclient)

        return bqclient

    except Exception as err:
        # Log error with exception details
        logging.error('Failed to create BigQuery client.', exc_info=True)
        return None


# In[ ]:


def create_bigquery_client(logging=None):
    """
    Creates a BigQuery client.
    Optionally, log any relevant information if logging is passed.
    """
    from google.cloud import bigquery

    if logging:
        logging.info("Creating BigQuery client...")

    # Initialize the BigQuery client
    client = bigquery.Client()

    return client


# In[ ]:


bqclient = create_bigquery_client()


# In[ ]:


def upload_bigquery_table(logging, bqclient, table_path, write_disposition, df):
    """
    upload_bigquery_table
    Accepts a path to a BigQuery table, the write disposition and a dataframe
    Loads the data into the BigQuery table from the dataframe.
    for credentials.
    The write disposition is either
    write_disposition="WRITE_TRUNCATE"  Erase the target data and load all new data.
    write_disposition="WRITE_APPEND"    Append to the existing table
    """
    try:
        logging.info('Creating BigQuery Job configuration with write_disposition=%s', write_disposition)
        job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
        logging.info('Submitting the BigQuery job')
        job = bqclient.load_table_from_dataframe(df, table_path, job_config=job_config)
        logging.info('Job  results: %s',job.result())
    except Exception as err:
        logging.error('Failed to load BigQuery Table. %s', err)


# In[ ]:


def bigquery_table_exists(bqclient, table_path):
    """
    bigquery_table_exists
    Accepts a path to a BigQuery table
    Checks if the BigQuery table exists.
    Returns True or False
    """
    try:
        bqclient.get_table(table_path)
        return True
    except NotFound:
        return False


# In[ ]:


def query_bigquery_table(logging, table_path, bqclient, surrogate_key):
    """
    query_bigquery_table
    Accepts a path to a BigQuery table and the name of the surrogate key
    Queries the BigQuery table but leaves out the update_timestamp and surrogate key columns
    Returns the dataframe
    """
    bq_df = pd.DataFrame
    sql_query = 'SELECT * EXCEPT ( update_timestamp, ' + surrogate_key + ') FROM `' + table_path + '`'
    logging.info('Running query: %s', sql_query)
    try:
        bq_df = bqclient.query(sql_query).to_dataframe()
    except Exception as err:
        logging.info('Error querying the table. %s', err)
    return bq_df


# In[ ]:


def add_surrogate_key(df, dimension_name, offset=1):
    """
    add_surrogate_key
    Accepts a data frame and inserts an integer identifier as the first column
    Returns the modified dataframe
    """
    df.reset_index(drop=True, inplace=True)
    df.insert(0, dimension_name + '_dim_id', df.index + offset)
    return df


# In[ ]:


def add_update_date(df, current_date):
    """
    add_update_date
    Accepts a data frame and inserts the current date as a new field
    Returns the modified dataframe
    """
    df['update_date'] = pd.to_datetime(current_date)
    return df


# In[ ]:


def add_update_timestamp(df):
    """
    add_update_timestamp
    Accepts a data frame and inserts the current datetime as a new field
    Returns the modified dataframe
    """
    df['update_timestamp'] = pd.Timestamp('now', tz='utc').replace(microsecond=0)
    return df


# In[ ]:


def build_new_table(logging, bqclient, dimension_table_path, dimension_name, df):
    """
    build_new_table
    Accepts a path to a dimensional table, the dimension name and a data frame
    Add the surrogate key and a record timestamp to the data frame
    Inserts the contents of the dataframe to the dimensional table.
    """
    logging.info('Target dimension table %s does not exit', dimension_table_path)

    if df is not None and not df.empty:
        df = add_surrogate_key(df, dimension_name, 1)
        df = add_update_timestamp(df)
        upload_bigquery_table(logging, bqclient, dimension_table_path, 'WRITE_TRUNCATE', df)
    else:
        logging.warning('No data to insert into the new table.')


# In[ ]:


def insert_existing_table(logging, bqclient, dimension_table_path, dimension_name, surrogate_key, df):
    """
    insert_existing_table
    Accepts a path to a dimensional table, the dimension name and a data frame
    Compares the new data to the existing data in the table.
    Inserts the new/modified records to the existing table
    """
    bq_df = pd.DataFrame()
    logging.info('Target dimension table %s exits. Checking for differences.', dimension_table_path)
    bq_df = query_bigquery_table(logging, dimension_table_path, bqclient, surrogate_key)
    new_records_df = df[~df.apply(tuple,1).isin(bq_df.apply(tuple,1))]
    logging.info('Found %d new records.', new_records_df.shape[0])
    if new_records_df.shape[0] > 0:
        new_surrogate_key_value = bq_df.shape[0] + 1
        new_records_df = add_surrogate_key(new_records_df, dimension_name, new_surrogate_key_value)
        new_records_df = add_update_timestamp(new_records_df)
        upload_bigquery_table(logging, bqclient, dimension_table_path, 'WRITE_APPEND', new_records_df)


# **Dimension Tables**

# **311 Taxi Complaints Dataset**

# In[ ]:


import os
# Check if the file exists
print(os.path.exists('/content/drive/My Drive/CIS9350/311_Taxi_Complaint.csv'))


# In[ ]:


#gcp_project = 'my-project-0577-434418'
#bq_dataset = '311_taxi_complaints'
#dimension_table_path = f'{gcp_project}.taxi_and_restaurant_dataset.{table_name}'


# In[ ]:


df1 = pd.read_csv('311_Taxi_Complaint.csv')
print(df1.columns)


# In[ ]:


#ls /content/drive/My\ Drive/CIS9350/


# In[ ]:


import os
import logging
import pandas as pd
import gdown
from datetime import datetime
from google.cloud import bigquery  # Make sure the BigQuery client library is installed


# In[ ]:


# Constants for GCP and file paths
gcp_project = 'my-project-0577-434418'
bq_dataset = '311_taxi_complaints'
csv_file_path = '/content/drive/My Drive/CIS9350/311_Taxi_Complaint.csv'
log_file_dir = '/content/drive/My Drive/CIS9350/logs/'


# In[ ]:


# Dimension dictionary for Taxi Complaints
dim_dict = {
    'complaint_type': ['complaint_type', 'descriptor'],
    'location': ['location_type', 'street_name', 'city', 'borough', 'incident_zip', 'longitude', 'latitude'],
    'agency': ['agency', 'agency_name'],
    'complaints_status': ['status'],
}


# In[ ]:


# Main ETL process for each dimension
def process_etl(logging, dimension_name, file_path, columns, dimension_table_path):
    """ETL process for a specific dimension."""
    try:
        # Initialize the dataframe
        df = pd.DataFrame()

        #df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]

        # Load and transform data
        df = load_csv_data_file(logging, file_path, df)

        if df.empty:
            logging.warning(f"No data found in {file_path}. Skipping dimension {dimension_name}.")
            return

        df = transform_data(logging, columns, df)

        # Create BigQuery client
        bqclient = create_bigquery_client(logging)
        target_table_exists = bigquery_table_exists(bqclient, dimension_table_path)

        # Load data into BigQuery
        if not target_table_exists:
            build_new_table(logging, bqclient, dimension_table_path, df)
        else:
            insert_existing_table(logging, bqclient, dimension_table_path, df)

        logging.info(f"Successfully processed dimension {dimension_name}.")
    except Exception as e:
        logging.error(f"Error processing dimension {dimension_name}: {e}")

# Main loop to run ETL for all dimensions
def run_etl():
    for key, value in dim_dict.items():
        dimension_name = key
        columns = value
        table_name = f'{dimension_name}_dimension'
        dimension_table_path = f'{gcp_project}.{bq_dataset}.{table_name}'

        # Configure logging
        #configure_logging(dimension_name)

        # Process the data for the specific dimension
        process_etl(logging, dimension_name, csv_file_path, columns, dimension_table_path)

        logging.info(f"ETL process completed for dimension {dimension_name}.")
    logging.shutdown()

# Run the ETL process
if __name__ == "__main__":
    run_etl()


# In[ ]:


import os

# Directory where the logs are stored
log_directory = '/content/drive/My Drive/CIS9350/logs/'

# List all log files
log_files = [f for f in os.listdir(log_directory) if f.endswith('.log')]

# Print file details (simulating 'ls -l')
for log_file in log_files:
    file_path = os.path.join(log_directory, log_file)
    file_stats = os.stat(file_path)
    print(f"{log_file} - Size: {file_stats.st_size} bytes - Last Modified: {datetime.fromtimestamp(file_stats.st_mtime)}")


# In[ ]:


get_ipython().system('tail -35 "/content/drive/My Drive/CIS9350/logs/etl_complaint_type_20241126.log"')


# In[ ]:


get_ipython().system('tail -35 "/content/drive/My Drive/CIS9350/logs/etl_location_20241126.log"')


# **Restaurant Inspection**

# In[ ]:


df = pd.read_csv('Restaurant_Inspection.csv')
print(df.columns)


# In[ ]:


# Constants for GCP and file paths
gcp_project = 'my-project-0577-434418'
bq_dataset = 'restaurant_complaints'
csv_file_path = '/content/drive/My Drive/CIS9350/Restaurant_Inspection.csv'
log_file_dir = '/content/drive/My Drive/CIS9350/logs/'


# In[ ]:


import os
import logging
import pandas as pd
from datetime import datetime


# Dimension dictionary
dim_dict = {
    'inspection_type': ['inspection_type'],
    'location': ['building', 'street', 'boro', 'zipcode', 'longitude', 'latitude'],
    'grade': ['grade', 'score'],
    'violation_code': ['violation_code', 'violation_description'],
    'critical_flag': ['critical_flag'],
    'restaurant_type': ['dba', 'cuisine_description'],
}


# In[ ]:


#def map_grade(score):
 #   if score >= 90:
  #      return 'A', 'Excellent'
   # elif score >= 80:
    #    return 'B', 'Good'
    #elif score >= 70:
     #   return 'C', 'Fair'
    #else:
     #   return 'P', 'Pending'  # Or 'Z' based on specific conditions

# Add the grade category and description
#df['Grade'], df['Grade_Description'] = zip(*df['score'].apply(map_grade))


# In[ ]:


import os
import logging
import pandas as pd
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account


# Function to create BigQuery client
def create_bigquery_client():
    credentials = service_account.Credentials.from_service_account_file('/content/drive/My Drive/CIS9350/my-project-0577-434418-59d2bed1a85c.json')
    return bigquery.Client(credentials=credentials, project=gcp_project)

# Function to check if BigQuery table exists
def bigquery_table_exists(client, table_path):
    try:
        client.get_table(table_path)
        return True
    except Exception as e:
        return False

# Function to build new table in BigQuery
def build_new_table(logging, client, table_path, df):
    try:
        logging.info(f"Building new table {table_path}")
        schema = []
        for column in df.columns:
            schema.append(bigquery.SchemaField(column, "STRING"))

        table = bigquery.Table(table_path, schema=schema)
        table = client.create_table(table)  # Create table
        logging.info(f"Table {table_path} created successfully.")

        # Load data into the new table
        load_data_to_bigquery(client, table_path, df)

    except Exception as e:
        logging.error(f"Error building new table {table_path}: {e}")

# Function to insert data into existing BigQuery table
def insert_existing_table(logging, client, table_path, df):
    try:
        logging.info(f"Inserting data into existing table {table_path}")
        load_data_to_bigquery(client, table_path, df)

    except Exception as e:
        logging.error(f"Error inserting data into {table_path}: {e}")

# Function to load data into BigQuery
def load_data_to_bigquery(client, table_path, df):
    try:
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",  # Append data to existing table
            source_format=bigquery.SourceFormat.CSV,
        )
        job = client.load_table_from_dataframe(df, table_path, job_config=job_config)
        job.result()  # Wait for the job to complete
        logging.info(f"Data successfully loaded into {table_path}")
    except Exception as e:
        logging.error(f"Error loading data to BigQuery: {e}")

# Function to transform data
def transform_data(columns, df):
    df = df[columns]  # Select only the required columns
    return df

# Main ETL function
def process_etl(logging, dimension_name, file_path, columns, dimension_table_path):
    """ETL process for a specific dimension."""
    try:
        # Initialize the dataframe
        df = pd.read_csv(file_path)

        if df.empty:
            logging.warning(f"No data found in {file_path}. Skipping dimension {dimension_name}.")
            return

        # Transform data
        df = transform_data(columns, df)

        # Add specific transformations for 'location'
        if dimension_name == 'location':
            df['state'] = 'NY'

        # Create BigQuery client
        bq_client = create_bigquery_client()
        target_table_exists = bigquery_table_exists(bq_client, dimension_table_path)

        # Load data into BigQuery
        if not target_table_exists:
            build_new_table(logging, bq_client, dimension_table_path, df)
        else:
            insert_existing_table(logging, bq_client, dimension_table_path, df)

        logging.info(f"Successfully processed dimension {dimension_name}.")

    except Exception as e:
        logging.error(f"Error processing dimension {dimension_name}: {e}")

# Main loop for processing all dimensions
for key, value in dim_dict.items():
    dimension_name = key
    table_name = f'{dimension_name}_dimension'
    dimension_table_path = f'{gcp_project}.{bq_dataset}.{table_name}'
    columns = value

    # Configure logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    current_date = datetime.today().strftime('%Y%m%d')
    log_filename = f'{log_file_dir}etl_{dimension_name}_{current_date}.log'
    logging.basicConfig(filename=log_filename, encoding='utf-8', format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('=========================================================================')
    logging.info(f'Starting ETL Run for dimension {dimension_name} on date {current_date}')

    # Process data
    process_etl(logging, dimension_name, csv_file_path, columns, dimension_table_path)

    logging.info(f"ETL process completed for dimension {dimension_name}.")
    logging.shutdown()


# In[ ]:


import os

# Directory where the logs are stored
log_directory = '/content/drive/My Drive/CIS9350/logs/'

# List all log files
log_files = [f for f in os.listdir(log_directory) if f.endswith('.log')]

# Print file details (simulating 'ls -l')
for log_file in log_files:
    file_path = os.path.join(log_directory, log_file)
    file_stats = os.stat(file_path)
    print(f"{log_file} - Size: {file_stats.st_size} bytes - Last Modified: {datetime.fromtimestamp(file_stats.st_mtime)}")


# In[ ]:


get_ipython().system('tail -35 "/content/drive/My Drive/CIS9350/logs/etl_location_20241126.log"')


# In[ ]:


datasets = client.list_datasets()  # List all datasets in the project
for dataset in datasets:
    print(f"Dataset: {dataset.dataset_id}")


# In[ ]:


from google.cloud import bigquery
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    '/content/drive/My Drive/CIS9350/my-project-0577-434418-59d2bed1a85c.json'
)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
print("BigQuery client created successfully.")


# In[ ]:


client.create_dataset('311_complaints_dataset', exists_ok=True)


# In[ ]:


client.load_table_from_dataframe(df_transformed, table_ref)  # Loading data into BigQuery table


# In[ ]:


import os
import logging
from datetime import datetime


# Get the current date
current_date = datetime.today().strftime('%Y%m%d')

# List all log files in the current directory
logs = [f for f in os.listdir() if f.endswith('.log') and f.startswith('etl_')]

# If log files are found, print the content of the first one
if logs:
    log_filename = logs[0]  # Select the first log file
    print(f"Reading log file: {log_filename}")
    with open(log_filename, 'r') as log_file:
        print(log_file.read())
else:
    print("No log files found in the current directory.")


# In[1]:


get_ipython().system('jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --to notebook --inplace Group3.ipynb')


# In[ ]:




