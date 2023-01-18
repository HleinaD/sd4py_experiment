# %%
import numpy as np
import pandas as pd
import re
import streamlit as st
import os 
import matplotlib.pyplot as plt 
import sd4py
import sd4py_extra
import warnings
import io
import copy
import datetime
import pickle
import scipy.io


# %%


# %%

def get_img_array_bytes(fig):

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=150)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=150)
    io_buf.seek(0)
    img_bytes = io_buf.getvalue()
    io_buf.close()

    return img_arr, img_bytes

# %%
def return_EPA():

    st.title('Exploratory Pattern Analytics (EPA)')

    st.markdown(
    '''
    Subgroup discovery is a technique that identifies clusters of data 
    within a dataset that can be defined by an understandable description. 
    Each description is a combination of logical rules, for example: 
    `Property_A > 5, AND, Property_B < 2`. 
    Every subgroup has a description like this, and can be used to identify a set of points 
    within a dataset (by applying the rules to the data). 

    Now, you will be asked to perform a task involving subgroup discovery. 
    You will be shown a list of different subgroups, 
    plus information about the subgroups, 
    and then will be asked to select the subgroup that you think is best for the task. 
    You will be given examples of how to calculate a score to rate 
    the performance of a subgroup. 
    After submitting your choice of three subgroups, 
    you will be given a score for the task. 
    Your score will be recorded as part of this study. 
    Your score will not count towards the grade you receive for any course 
    or study programme. Remember that your responses will be anonymised. 
    
    After performing the task, you will be asked to fill in an online 
    questionnaire about different ways of presenting information about subgroups, 
    where you will share your thoughts on how helpful they are. 
    
    To get started, please enter the pseudonymous code you have been given on your information sheet. 
    ''')

    pseudonym = st.text_input("Please enter your pseudonymous code:")

    if 'pseudonym_entered' not in st.session_state:
        st.session_state['pseudonym_entered'] = False
    
    pseudonym_entered_button = st.button("Continue")

    if pseudonym_entered_button:
        st.session_state['pseudonym_entered'] = True

    code_good = False

    if st.session_state['pseudonym_entered']:
        regex_match = re.match("([BE])[0-9]{3}([BE])", pseudonym.strip().upper())

        assert regex_match is not None, "Code was not recognised, please check your pseudonymous code is entered correctly, or try re-entering the code."

        first_letter = regex_match.group(1)
        last_letter = regex_match.group(2)
        
        assert first_letter == last_letter, "Code was not recognised, please check your pseudonymous code is entered correctly, or try re-entering the code."

        code_good = True

    if not code_good:

        st.stop()

    st.markdown("Thank you!")

    @st.cache
    def get_data():

        credit_g = scipy.io.arff.loadarff('/home/daniel/Downloads/dataset_31_credit-g.arff')
        credit_g = pd.DataFrame.from_records(credit_g[0])

        for col in credit_g:

            try:

                credit_g[col] = credit_g[col].str.decode('utf-8')

            except AttributeError:

                pass

        credit_g['class'] = credit_g['class'] + '_loan'

        np.random.seed(42)
        randomised_index = np.random.choice(np.arange(1000), 1000)
        credit_g_train = credit_g.iloc[randomised_index[:250]]
        credit_g_val = credit_g.iloc[randomised_index[250:500]]
        credit_g_test = credit_g.iloc[randomised_index[500:]]

        return credit_g_train, credit_g_val, credit_g_test

    train, validation, test = get_data()

    st.markdown(
    '''
    ## The task

    Below, you will see subgroups used to analyse data from the Covid-19 (coronavirus) pandemic.

    The data being analysed is as folows. 
    Each data point in the dataset represents a single month of data for some country,
    and includes information like the average number of cases per day and the average 
    number of hospital admissions. 
    The subgroups all attempt to distinguish months belonging to the year 2022 (the most recent year) from 
    months belonging to 2020 (when the pandemic first became global). Specifically, the subgroups
    aim to find a description of points that belong to 2022. 
    You will see subgroups that have been trained on a small subset of training data, 
    and will need to decide which ones you believe will perform well 
    on new data. 
    Please note there is no 'ideal' subgroup that perfectly separates months belonging to
    2020 from months belonging to 2022.

    To complete the task, you will choose the subgroup that you think will perform best,
    which will then be applied to  
    a previously-unseen 'test set' of data points. After submitting your choice,
    you will also be shown a score for how well the subgroup performs the task. 
    The score will simply be the number of points (in the test set) identified by the subgroup 
    that belong to the year 2022, minus the number of points from 2020 identified by the subgroup. 
    Therefore, selecting many points with a high precision will give a high score. 
    '''
    )

    st.markdown(
    '''
    ## Top subgroups

    The table of results shows a list of the best subgroups found, along with some measures of quality. 
    Each subgroup involves up to three variables, and includes a rule for each of those variables. 
    These rules combine to select points within the dataset. 
    '''
    )

    @st.cache
    def get_subgroups():

        with open('/home/daniel/Documents/private-EPA-2/experiment_drafting/credit_g_subgroups.pkl', 'rb') as f:
            subgroup_list = pickle.load(f)

        subgroups = sd4py.PySubgroupResults(
            subgroup_list, 
            1, 
            1, 
            'class', 
            'bad_loan'
        )

        return subgroups[:10]

    subgroups = get_subgroups()


    ## To make the subgroup names more readable
    ids = ['*A*', '*B*', '*C*', '*D*', '*E*', '*F*', '*G*', '*H*', '*I*', '*J*', '*K*', '*L*', '*M*'][:len(subgroups)]
    labels = [re.sub('AND', '\nAND',str(key)) for key in subgroups]
    labels = ['({}) {}'.format(*vals) for vals in zip(ids, labels)]



    if first_letter == 'B':

        st.markdown(
        '''
        This table of results shows a list of subgroups, along with the size and precision 
        (what proportion of the points selected by the pattern in fact belong to the target group) that were achieved
        by the subgroups on the training data. 
        '''
        )

        st.table(subgroups.to_df().drop(columns='quality'))

    else:

        st.markdown(
        '''
        This table of results shows a list of subgroups, along with some measures of quality calculated against a validation data set 
        (not the data originally used to discover subgroups). 
        The percentage of subgroup members that belong to the target class is shown. 
        This number, along with the size of the subgroup, is used to calculate the quality score of the pattern. 
        For extra information, the precision (what proportion of the points selected by the pattern in fact belong to the target group), 
        the recall (how much of the target group is selected by the pattern), 
        and the F1-score (a combination of precision and recall) are provided as extra quality measures. 
        For example, a precision of 0.6 means that 60% of the data points selected by the subgroup belong to the target class.  
        Estimated 5% and 95% confidence intervals are shown for precision, recall and F-1.
        '''
        )

        target = 'class'
        target_value = 'bad_loan'
        target_nominal = True

        @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
        def get_bootstrap():

            frac = 1.0

            if len(validation) > 13747: ## 13747 / log_2(l3747) = 1000

                frac = 1 / np.log2(len(validation))

            else:

                frac = min(frac, 1000 / len(validation))

            if target_nominal:

                subgroups_bootstrap = subgroups.to_df().merge(
                    sd4py_extra.confidence_precision_recall_f1(subgroups, 
                                                            validation, 
                                                            number_simulations=100,
                                                            frac=frac
                                                            )[1], 
                    on="pattern")

                subgroups_bootstrap = subgroups_bootstrap#.sort_values('f1_lower', ascending=False)

            else:

                subgroups_bootstrap = subgroups.to_df().merge(
                    sd4py_extra.confidence_hedges_g(subgroups, 
                                                    validation, 
                                                    number_simulations=100)[1], 
                    on="pattern")

                subgroups_bootstrap = subgroups_bootstrap#.sort_values('hedges_g_lower', ascending=False)

            return subgroups_bootstrap

        subgroups_bootstrap = get_bootstrap()

        #subgroups_bootstrap_top10, subgroups_selection, ids = get_top10_subgroups_selection_ids()
        subgroups_bootstrap_top10 = subgroups_bootstrap.copy().rename(columns={
                    'pattern':'Pattern',
                    'size':'Size',
                    'quality':'Quality Score',
                    'target_evaluation':'% of Subgroup that Are Target Class',
                    'precision_lower':'Precision (lower CI)',
                    'precision_upper':'Precision (upper CI)',
                    'recall_lower':'Recall (lower CI)',
                    'recall_upper':'Recall (upper CI)',
                    'f1_lower':'F-1 Score (lower CI)',
                    'f1_upper':'F-1 Score (upper CI)'
                })
        
        subgroups_bootstrap_top10['% of Subgroup that Are Target Class'] = subgroups_bootstrap_top10['% of Subgroup that Are Target Class'] * 100

        subgroups_bootstrap_top10.insert(0, 'id', ids)
        subgroups_selection = subgroups

        st.table(subgroups_bootstrap_top10.astype({'Size':int}).style.set_precision(2))


        st.markdown(
        '''
        ## Plotting the distribution of the target value 

        This visualisation shows the expected variability for the target value, 
        meaning how much it changes across different samples of measurements (taken from a validation data set). 
        'Target Value' means the proportion of points selected by the subgroup that belong to the target class. 

        This is depicted through boxes in a box plot, with wider boxes in the (horizontal) x-direction implying greater variability. 
        The orange line shows the target value on average across different samples. 
        How many points are selected by each pattern is also shown (i.e., its size), 
        with thicker/taller boxes in the vertical direction meaning that a pattern selects a greater number of points on average.
        '''
        )

        @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
        def get_conf_int():
                
            with warnings.catch_warnings():

                warnings.simplefilter("ignore")
                
                return sd4py_extra.confidence_intervals(subgroups_selection, validation)

        results_dict, aggregation_dict = get_conf_int()

        @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
        def get_boxplots():

            results_list = [results_dict[name] for name in subgroups_bootstrap_top10['Pattern']]

            fig = plt.figure(dpi = 150)
            
            sd4py_extra.confidence_intervals_to_boxplots(results_list[::-1], labels=labels[::-1])  ## Display is backwards by default

            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            #plt.xlabel('Proportion of Subgroup Members that Had Fault within 30 Minutes', size=12)
            plt.gca().set_title('Distribution of ' + str(target) + ' from Bootstrapping',pad=20)
            fig.set_size_inches(17,10)
            plt.tight_layout()

            ## Convert to image to display 

            return get_img_array_bytes(fig)

        img_arr, img_bytes = get_boxplots()

        st.image(img_arr)

        st.download_button('Save boxplots', img_bytes, file_name="{}_boxplots.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), mime="image/png")

        st.markdown(
        '''
        ## Overlap between subgroups

        After discovering subgroups, it is possible that two different subgroups 
        might essentially be different ways of describing the same points in the data. 
        In this case, it might be useful to know that they are closely related. 

        On the other hand, subgroups might have an extreme target value for different reasons.
        If two subgroups select quite different data points, then there might be different reasons they are interesting, 
        and it could be worthwhile to investigate them both separately in greater detail.

        In this visualisation, subgroups are connected to each other by how much they overlap.
        If two subgroups select similar subsets of data, 
        then they have a strong link between them and appear closer together. 
        Overall, this visualisation takes the form of a network diagram.
        '''
        )

        edges_threshold = st.slider("Only draw edges when overlap is greater than: ", 0.0, 1.0, 0.25)

        @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
        def get_jaccard_plot():

            fig = plt.figure(dpi=150)

            sd4py_extra.jaccard_visualisation(subgroups_selection, 
                                                validation, 
                                                edges_threshold, 
                                                labels=labels)

            fig.set_size_inches(20,9)
            plt.margins(x=0.15)
            plt.gca().set_frame_on(False)
            plt.gca().set_title('Jaccard Similarity Between Subgroups', fontsize=14)
            fig.tight_layout()

            ## Convert to image to display 

            return get_img_array_bytes(fig)

        img_arr, img_bytes = get_jaccard_plot()

        st.image(img_arr)

        st.download_button('Save network diagram', img_bytes, file_name="{}_network_diagram.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), mime="image/png")

        st.markdown(
        '''
        ## Focus on a specific subgroup

        The next visualisation makes it possible to examine subgroups that are particularly interesting in more detail. 
        This visualisation compares subgroup members (points selected by the subgroup) to non-members 
        (these non-members are also known as the 'complement') for one specific subgroup. 

        The target variable, the variables used to define the subgroup (selector variables), 
        and additional variables that are most clearly different between members and non-members are shown. 
        These respectively appear in the top-left, top-right and bottom panels of the visualisation. 
        This makes it possible to see additional information about the subgroup, 
        and understand more about the circumstances in which the subgroup occurs. 

        In the top-left, the distribution of values for the target variable is shown. 
        A different set of horizontal boxes is used for the subgroup and the complement. 
        In the remaining panels, the subgroup is also indicated by a solid blue line and the complement by a dashed orange line. 
        '''
        )

        chosen_sg_options = copy.deepcopy(labels)
        chosen_sg_options.insert(0, 'Choose a subgroup to visualise in more detail')
        chosen_sg = st.selectbox('Subgroup to focus on: ', chosen_sg_options)

        if chosen_sg != 'Choose a subgroup to visualise in more detail':

            chosen_sg = subgroups_selection[dict(zip(labels, list(range(10))))[chosen_sg]]

            saved_figsize = plt.rcParams["figure.figsize"]

            @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
            def get_subgroup_overview():

                plt.rcParams["figure.figsize"] = (20,17)

                fig = plt.figure(dpi = 150)
                fig.suptitle(re.sub('AND', '\nAND',str(chosen_sg)), y=0.95)
                plt.tight_layout()
                sd4py_extra.subgroup_overview(chosen_sg, validation, axis_padding=50)

                ## Convert to image to display - so that Streamlit doesn't try to resize disasterously. 

                return get_img_array_bytes(fig)

            img_arr, img_bytes = get_subgroup_overview()

            st.image(img_arr)

            st.download_button('Save subgroup overview', img_bytes, file_name="{}_subgroup_overview.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), mime="image/png")

    # plt.rcParams["figure.figsize"] = saved_figsize

    # if not isinstance(validation.index, pd.DatetimeIndex):

    #     st.stop()
    
    # if analysis_type != 'Event detection':

    #     st.stop()

    # st.markdown(
    # '''
    # ## Specific subgroup members

    # Finally, if the data comes from a process that happens over time, we can focus on particular moments at which a pattern occurs, 
    # to see what happens to different variables before, during, and after. 
    # After selecting a single pattern, you can now select a particular moment when the pattern occurs, 
    # from the drop-down list below. 
    # The target variable is shown, along with the other variables that are most clearly different between subgroup members 
    # and non-members. The moment at which the pattern occurs is indicated by a red rectangle in the background. 
    # '''
    # )

    # chosen_member_options = copy.deepcopy(chosen_sg.get_rows(dataset_production).index.tolist())
    # chosen_member_options.insert(0, 'Choose a subgroup member to inspect')
    # chosen_member = st.selectbox('Subgroup member to inspect: ', chosen_member_options)

    # if chosen_member == 'Choose a subgroup member to inspect':
    #     st.stop()

    # before = st.number_input("Also display earlier time points that happened within: ", step=1, value=10, min_value=1)
    # after = st.number_input("Also display later time points that happened within: ", step=1, value=10, min_value=1)

    # before_after_unit = st.selectbox(
    #     "Unit of time:",
    #     ["", "Hours", "Minutes", "Seconds", "Milliseconds"],
    #     key='before_after_unit'
    # )

    # if before_after_unit == '':
    #     st.stop()

    # @st.cache(hash_funcs={pd.DataFrame: id, sd4py.PySubgroupResults:id})
    # def get_most_interesting():

    #     most_interesting_numeric = sd4py_extra.most_interesting_columns(chosen_sg, dataset_production.drop(columns=chosen_sg.target))[0][:7]

    #     return most_interesting_numeric.index

    # most_interesting = get_most_interesting()

    # fig = plt.figure(dpi = 150)

    # start_time = chosen_member-pd.Timedelta(before, unit=before_after_unit.lower())
    # end_time = chosen_member+pd.Timedelta(after, unit=before_after_unit.lower())

    # iidx = dataset_production.index.get_loc(chosen_member)

    # if iidx > 0: 
    #     previous_time = dataset_production.index[iidx - 1]
    #     if previous_time < start_time:
    #         start_time = previous_time
    # if iidx < (len(dataset_production) - 1):
    #     next_time = dataset_production.index[iidx + 1]
    #     if next_time > end_time:
    #         end_time = next_time 

    # sd4py_extra.time_plot(chosen_sg, dataset_production.loc[start_time:end_time], 
    #     dataset_production[target].loc[start_time:end_time],
    #     *[dataset_production[col].loc[start_time:end_time] for col in most_interesting],
    #     window_size=1, use_start=True)

    # fig.suptitle('Variables over time for ({})'.format(str(chosen_sg)), y=1.0, size =14)    

    # fig.set_size_inches(18,20)
    # plt.tight_layout()

    # ## Convert to image to display

    # img_arr, img_bytes = get_img_array_bytes(fig)

    # st.image(img_arr)

    # st.download_button('Save member time plot', img_bytes,
    #     file_name="{}_time_plot_member_{}.png".format(
    #         datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
    #         '_'.join(str(dataset_production.index[iidx]).strip().split(' '))), 
    #     mime="image/png")

    st.markdown('''
    ## Submit answer

    If you feel ready to choose a subgroup that you think best performs the task, 
    then please do so now. 
    Please select your preferred subgroup from the drop-down list below, 
    and then click the button labelled 'Submit'.
    Afterwards, you will be shown a score, and a hyperlink to an online questionnaire. 
    Please navigate to the hyperlink and fill in the questionnaire. 
    ''')

    def get_score(answer_sg):

        rows = answer_sg.get_rows(test)

        return (3 * (rows[target] == target_value).sum()) - ((rows[target] != target_value).sum())
    
    answer_sg_options = copy.deepcopy(labels)
    answer_sg_options.insert(0, 'Choose the subgroup you think performs the task the best')
    answer_sg = st.selectbox('I think the best subgroup is: ', answer_sg_options)
    
    if answer_sg == 'Choose the subgroup you think performs the task the best':

        st.stop()

    if 'answer_submitted' not in st.session_state:

        st.session_state['answer_submitted'] = False
    
    answer_submitted_button = st.button("Submit")

    if answer_submitted_button:
        st.session_state['answer_submitted'] = True
    
    if not st.session_state['answer_submitted']:
        st.stop()

    st.markdown('This subgroup achieved a score of {}'.format(get_score(subgroups_selection[dict(zip(labels, list(range(10))))[answer_sg]])))

    st.markdown(
    '''Please navigate to the following URL and answer the follow-on questionnaire: <https://www.survey.uni-osnabrueck.de/limesurvey/index.php/443113?Pseudonym={}>. 
    '''.format(pseudonym)
    )

return_EPA()


# %%
