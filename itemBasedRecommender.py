import pandas as pd
import numpy as np
import scipy.sparse as sciSp
import dataMiner


class ItemBasedRecommender():
    def __init__(self, env='None',
                 data_source='CSV',
                 corr_method='pearson',
                 rating_method='penalized',
                 tables_type='numpy',
                 field_weight={
                         'preview': 0.5,
                         'view': 1.0,
                         'save': 1.0,
                         'download': 3.0,
                         'forward': 5.0,
                         'share': 5.0,
                         'rating': 1.0},
                 min_period=1):
        if corr_method not in {'pearson', 'kendall', 'spearman'}:
            raise Exception('This correlation method is not supported by Pandas. \n'
                            'Please choose one of the followings: pearson, kendall, spearman.')

        if rating_method not in {'penalized', 'normal'}:
            raise Exception('This method is not supported by itemBasedRecommender for Ratings.'
                            'Please choose from the followings: penalized, normal.')

        if data_source not in {'DB', 'CSV'}:
            raise Exception('This data_source is not define for itemBasedRecommender.'
                            'Please choose from the followings: DB, CSV.')
        elif data_source == 'DB' and env is None:
            raise Exception('None environement is not acceptable.')
        self.corr_method = corr_method
        self.min_period = min_period
        self.field_weight = field_weight
        self.rating_method = rating_method
        self.tables_type = tables_type
        self.DM = dataMiner.dataMiner(env=env,
                                      data_source=data_source,
                                      fields=field_weight,
                                      tablesType=self.tables_type)
        self.user_content_activity = self.DM.get_user_content_activity_info()
        self.user_content_table = dict((x, []) for (x, y) in field_weight.items() if y > 0.0)
        self.content_content_corr = dict((x, []) for (x, y) in field_weight.items() if y > 0.0)

        if self.tables_type == 'numpy':
            self.users_map, self.users_map_rev, \
            self.content_map, self.content_map_rev = \
                self.DM.get_users_and_contents_map_info_for_matrices()

    def get_user_content_table(self):
        if self.user_content_activity.keys() != self.user_content_activity.keys():
            raise Exception('something is wrong here!!! please check me!')

        if self.tables_type == 'panda':
            for field in self.user_content_activity.keys():
                self.user_content_table[field] = pd.pivot_table(self.user_content_activity[field],
                                                                index='member_id',
                                                                columns='content_id',
                                                                values=field)
        elif self.tables_type == 'numpy':
            num_all_users = len(self.users_map.keys())
            num_all_contents = len(self.content_map.keys())

            for field in self.user_content_activity.keys():

                data = self.user_content_activity[field][:, 2]
                users_index = np.array(self.users_map[self.user_content_activity[field][:, 0]])
                contents_index = np.array(self.content_map[self.user_content_activity[field][:, 1]])

                self.user_content_table[field] = \
                    sciSp.coo_matrix((data, (users_index, contents_index)),
                                     shape=(num_all_users, num_all_contents))

    def get_content_content_corr(self):
        for field in self.content_content_corr.keys():
            self.content_content_corr[field] = \
                self.user_content_table[field].corr(method=self.corr_method,
                                                    min_periods=self.min_period)

    def get_sim_candidates_per_member(self, member_id):
        # print('looking for similar candidates for member#{}'.format(member_id))
        all_candids = None
        for field in self.user_content_activity.keys():
            # print(self.user_content_table[field].index)
            if member_id in self.user_content_table[field].index:
                my_activity = self.user_content_table[field].loc[member_id].dropna()
                my_value_corrector = self.field_weight[field]
                for c_id, value in my_activity.items():
                    if field == 'rating' and self.rating_method == 'penalized':
                        my_value_corrector *= (value - 2.5)
                    value *= my_value_corrector

                    my_sim_cands = self.content_content_corr[field][c_id].dropna()
                    my_sim_cands *= my_sim_cands * value
                    if all_candids is None:
                        all_candids = my_sim_cands
                    else:
                        for cand, score in my_sim_cands.items():
                            try:
                                all_candids[cand] += score
                            except:
                                all_candids[cand] = score
        all_candids.sort_values(ascending=False, inplace=True)
        return all_candids

    def get_recommendations_for_member(self, member_id, num_arts=5):
        my_candids = self.get_sim_candidates_per_member(member_id)
        my_history = self.user_content_table['rating'].loc[member_id].dropna()
        my_recomms = my_candids[~my_candids.index.isin(my_history.index)]
        if len(my_recomms) < num_arts:
            return my_recomms
        else:
            return my_recomms[:num_arts]

if __name__ == "__main__":
    myFilter = ItemBasedRecommender(data_source='CSV', tables_type='panda')
    myFilter.get_user_content_table()
    myFilter.get_content_content_corr()
    recomms = myFilter.get_recommendations_for_member(194848)
    print(recomms)
