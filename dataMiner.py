import iniFileParser
#from newRepo.dbconnections import dbConnections
import csv
import pandas as pd
import numpy as np

rootDir = ''
query_ini_file_name = 'dataMinerQueries_orig.ini'
field_to_page_id = {'preview': 250,
                    'view': 251,
                    'save': 252,
                    'download': 254,
                    'forward': 255,
                    'share': 256}
class dataMiner():
    def __init__(self, env, data_source='DB',
                 save_queried_data=True,
                 tablesType='numpy',
                 fields={'preview': 0.5,
                         'view': 1.0,
                         'save': 1.0,
                         'download': 3.0,
                         'forward': 5.0,
                         'share': 5.0,
                         'rating': 1.0}):
        if data_source == 'DB':
            # self.db = dbConnection()
            pass
        else:
            self.db = None
        self.env = env
        self.fields = fields
        self.pandaOrNumpy = tablesType
        self.save_querid = save_queried_data
        self.data_source = data_source
        self.ini_parser = iniFileParser.iniFileParser(rootDir+query_ini_file_name)
        self.user_field_data = dict((key, []) for (key, value) in self.fields.items() if value > 0)

    def get_field_data_columns(self, field):
        if field == 'rating':
            return self.ini_parser.get_section_option(section='rating_field', option='columns').split(',')

        elif field != 'rating' and field in field_to_page_id.keys():
            return self.ini_parser.get_section_option(section='other_fields', option='columns').format(field).split(',')

        else:
            raise Exception('This option has not yet implemented.')

    def get_field_data_query(self, field):
        if field == 'rating':
            return self.ini_parser.get_section_option(section='rating_field', option='query')

        elif field != 'rating' and field in field_to_page_id.keys():
            return self.ini_parser.get_section_option(section='other_fields', option='query').format(field_to_page_id[field])

        else:
            raise Exception('This option has not yet implemented.')

    def get_user_content_activity_info(self):
        for field in self.fields.keys():
            if self.data_source in {'DB', 'CSV'}:
                csv_file_name = 'user_{}_data'.format(field)
                csv_file_path = rootDir + csv_file_name

                if self.data_source == 'DB':
                    query = self.get_field_data_query(field)
                    my_data = self.db.ReadData(query,
                                               self.env,
                                               dic_cursor_flag=False)
                    if self.save_querid:
                        with open(csv_file_path, 'wb') as csvFile:
                            my_writer = csv.writer(csvFile, delimiter=',', quotechar='|')
                            # my_writer.writerow(columns)
                            my_writer.writerows(my_data)

                if self.pandaOrNumpy == 'panda':
                    columns = self.get_field_data_columns(field)
                    field_data = pd.read_csv(csv_file_path, sep=',', names=columns, usecols=range(len(columns)))

                elif self.pandaOrNumpy == 'numpy':
                    field_data = np.genfromtxt(csv_file_path, delimiter=',')

            else:
                raise Exception('This option has not yet been implemented for get_user_activity method in dataMiner.'
                                ' Please choose one the followings: DB, CSV.')

            self.user_field_data[field] = field_data
        return self.user_field_data

    def get_users_and_contents_map_info_for_matrices(self):
        users_set = {}
        contents_set = {}
        for data in self.user_field_data.value():
            for row in data:
                users_set.add(row[0])
                contents_set.add(row[1])

        users_map = {data: index for index, data in enumerate(users_set)}
        users_map_rev = {y: x for (x, y) in users_map.items()}

        contents_map = {data: index for index, data in enumerate(contents_set)}
        contents_map_rev = {y: x for (x, y) in contents_map.items()}

        return users_map, users_map_rev, contents_map, contents_map_rev






