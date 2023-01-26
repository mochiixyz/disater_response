# Import library
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
 
def load_data(msg_filepath, ctg_filepath):
    """
    Load messages and categories data then merge 2 dataset
    
    Input:
        msg_filepath - File contain messages data
        ctg_filepath - File contain categories data
    Output:
        Dataframe combine messages and categories
    """
    
    #Read 2 dataset file
    msg = pd.read_csv(msg_filepath)
    ctg = pd.read_csv(ctg_filepath)

    #Combine 2 dataset
    df = pd.merge(msg,ctg,on='id')
    return df 

def clean_data(df):
    """
    Clean data
    
    Input:
        df - Combined data contain messages and categories
    Outputs:
        df - Cleaned Dataframe
    """
    
    #Split the categories
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Extract column names
    row = categories.head(1)
    col_names = row.applymap(lambda x: x[:-2]).iloc[0,:].tolist()
    categories.columns = col_names
    
    #Extract value of categories
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] =  categories[column].astype(int)
    
    categories.replace(2, 1, inplace=True)
    
    #Merge splited categories
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)

    #Drop dupe
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Input:
        df -> Cleaned Dataframe
        database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages_tbl', engine, if_exists = 'replace', index=False)


def main():
    """
    Main function start data processing. Steps are:
        1) Load Data (Messages and Categories data)
        2) Clean Data
        3) Save Data to SQLite Database
    """
    
    # Print step by step
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data ...')
        df = clean_data(df)
        
        print('Saving data... : {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets then the filepath of the database to save the cleaned data'\
              'Example: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
