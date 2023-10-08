import os
#Some globals used for parsing the directories etc.
directory_notation = '\\'

if os.name != 'nt':
    directory_notation = '/'
file_name_connector = '_'
csv_postfix ='.csv'


 
#Just openup the configuration dictionary. :)
def break_config_into_pieces_for_plots(config={}):
    assert config != {}
    return config['result_space'],config['classifier'],config['results_type'],config['optimizer_type'],config['seeds'],config['data_repo']

# A function that given a list of directories, concats them 
# in the order received.
#Returns the final directory location.
def parse_directory(list_of_directories = []):
    assert len(list_of_directories) != 0
    final_directory = list_of_directories[0]
    for i in list_of_directories[1:]:
        final_directory += directory_notation + i 
    return final_directory
