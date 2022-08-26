# So here I read a list of runs, that I copied manually from the terminal, and then find the top k runs for each model, and then do something with that
# The key part here is that each directory for the run has a "Validation-SELD-score.series.txt" file with the values for SELD score
#
# Copied from the Sony project

import os
import numpy as np
import pandas as pd
import argparse


def get_parameters():
    """
       Gets the parameters for the analysis.

    Mode:
        0 : Only reads the top k checkpoints and writes the file list_of_checkpoints.txt. Needed as first step for evaluation.
        1 : Computes the resutls for the tables, using the mean and stdev across seeds.
        2 : Computes teh results for the tables, using the
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3, 5], default=0,  help='Mode for the analysis.')
    parser.add_argument('--dcase2022', action='store_true', help="Enables DCASE22022 mode.")
    parser.add_argument('--filename', type=str, default='', help="suffix for the filenames, e.g. it will read 'list_of_checkpoints_filename.txt")
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials for the stratified bootstrap.')
    parser.add_argument('--chunk_id', type=int, default=0, help='Chunk id when running as part of an array job.')

    params = parser.parse_args()
    return params

def extract_top_checkpoints(root, dpath, top_k=3):
    dirs = os.listdir(dpath)

    # Find the top k iterations for the run
    for fname in dirs:
        if 'Validation-SELD-score.series.txt' in fname:
            try:
                df = pd.read_csv(os.path.join(dpath, fname), sep=" ", header=None)
                df.columns = ["iteration", "SELD_score"]

                best = df.nsmallest(top_k, 'SELD_score', keep='first')
                break
            except pd.errors.EmptyDataError:
                print(f'WARNING: Fname : {dpath} is empty')
                return

    print("========")
    print(f'fname = {dpath}')
    print(best)

    # Now we export the checkpoints to a text file
    output = []
    for fname in dirs:
        if '.pth' in fname:
            for iter in best['iteration']:
                if f'{iter:07}' in fname:
                    output.append(dpath + '/' + fname)

    return output

def run_evaluation(list_of_runs, num_classes=12, dcase2022=False):
    import pandas as dataframe
    from evaluation.dcase2022_metrics import cls_compute_seld_results, parameters
    params = parameters.get_params()

    # Compute just the DCASE 2021 final results
    if not dcase2022:
        score_obj = cls_compute_seld_results.ComputeSELDResults(params,
                                                                ref_files_folder="/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3/metadata_dev")
    else:
        score_obj = cls_compute_seld_results.ComputeSELDResults(params,
                                                                ref_files_folder="/m/triton/scratch/work/falconr1/sony/data_dcase2022/metadata_dev")
    print('')
    print('===== CSV format =====')
    print('run \t checkpoint \t ER \t F \t LE \t LR \t seld_score')

    df_MACRO = pd.DataFrame(columns=['group', 'seed', 'run', 'checkpoint', 'ER', 'F', 'LE', 'LR', 'seld_score', 'class_wise'])
    df_micro = pd.DataFrame(columns=['group', 'seed', 'run', 'checkpoint', 'ER', 'F', 'LE', 'LR', 'seld_score'])
    for run in list_of_runs:
        if '.pth' in run:
            tmp = run[0:-4]  # remove extension
        checkpoint = tmp[-7:]  # e.g. 0010000
        if not dcase2022:
            pred_output_format_files = os.path.join(os.path.dirname(run), f'pred_dcase2021t3_foa_devtest_{checkpoint}_sgl')  # Path of the DCASE output format files
        else:
            pred_output_format_files = os.path.join(os.path.dirname(run),
                                                    f'pred_dcase2022_devtest_all_{checkpoint}_sgl')  # Path of the DCASE output format files
        #ER, F, LE, LR, seld_scr, ER_S, ER_D, ER_I, prec, recall, L_prec, L_F = score_obj.get_SELD_Results(pred_output_format_files)
        seld_metric_macro, seld_metric_micro = score_obj.get_SELD_Results(pred_output_format_files, num_classes=num_classes)

        # Micro metrics
        ER, F, LE, LR, seld_scr, _ = seld_metric_micro
        # Printing in easy to parse format so that I can get the tables for the paper easily
        tmp = os.path.dirname(run).split('/')[-1]  # full run name
        if dcase2022:
            group = "".join(tmp.split('_')[3:])
            seed = tmp.split('_')[2]  # extracts sXXXX ---> XXXXX
        else:
            group = "".join(tmp.split('_')[2:])
            seed = tmp.split('_')[1]  # extracts sXXXX ---> XXXXX
        model = os.path.basename(run)
        result_str = 'micro: {}, {}, {}, {}, {}, {}, {}'.format(tmp, model, ER, F * 100, LE, LR * 100, seld_scr)
        print(result_str)
        df_micro = df_micro.append({'group': group, 'seed': seed, 'run': tmp, 'checkpoint': model,
                                    'ER': ER, 'F': F, 'LE': LE, 'LR': LR, 'seld_score': seld_scr}, ignore_index=True)

        # MACRO metrics
        ER, F, LE, LR, seld_scr, class_wise_scores = seld_metric_macro
#        # Printing in easy to parse format so that I can get the tables for the paper easily
#        tmp = os.path.dirname(run).split('/')[-1]
#        group = "".join(tmp.split('_')[2:])
#        seed = tmp.split('_')[1]  # extracts sXXXX ---> XXXXX
#        model = os.path.basename(run)
        result_str = 'MACRO: {}, {}, {}, {}, {}, {}, {}'.format(tmp, model, ER, F * 100, LE, LR * 100, seld_scr)
        print(result_str)
        df_MACRO = df_MACRO.append({'group': group, 'seed': seed, 'run': tmp, 'checkpoint': model,
                                    'ER': ER, 'F': F, 'LE': LE, 'LR': LR, 'seld_score': seld_scr, 'class_wise': class_wise_scores},
                                   ignore_index=True)
    return df_MACRO, df_micro

def aggregate_scores(df):
    print(df)

    #df = df.sort_values(['group', 'seed'])
    df_means = df.groupby(['group']).mean().reset_index(inplace=False)
    df_stds = df.groupby(['group']).std().reset_index(inplace=False)

    # NOTE: and then view the dataframes in the data view and copy the values manually to excel
    print(df_means)
    print(df_stds)
    return df_means, df_stds

def prepare_evaluation_bootstrap(list_of_runs, num_classes=12, dcase2022=False):
    import pandas as dataframe
    from evaluation.dcase2022_metrics import cls_compute_seld_results, parameters
    params = parameters.get_params()

    print('')
    print('===== CSV format =====')
    print('run \t checkpoint \t ER \t F \t LE \t LR \t seld_score')

    assert isinstance(list_of_runs, list), 'list_of_runs should be a list, not a string.'
    df = pd.DataFrame(columns=['group', 'seed', 'run', 'checkpoint', 'pred_path', 'ER', 'F', 'LE', 'LR', 'seld_score'])
    for run in list_of_runs:
        if '.pth' in run:
            tmp = run[0:-4]  # remove extension
        else:
            tmp = run
        checkpoint = tmp[-7:]  # e.g. 0010000
        if not dcase2022:
            pred_output_format_files = os.path.join(os.path.dirname(run), f'pred_dcase2021t3_foa_devtest_{checkpoint}_sgl')  # Path of the DCASE output format files
        else:
            pred_output_format_files = os.path.join(os.path.dirname(run), f'pred_dcase2022_devtest_all_{checkpoint}_sgl')  # Path of the DCASE output format files

        # Printing in easy to parse format so that I can get the tables for the paper easily
        tmp = os.path.dirname(run).split('/')[-1]  # full run name
        if dcase2022:
            group = "".join(tmp.split('_')[3:])
            seed = tmp.split('_')[2]  # extracts sXXXX ---> XXXXX
        else:
            group = "".join(tmp.split('_')[2:])
            seed = tmp.split('_')[1]  # extracts sXXXX ---> XXXXX
        model = os.path.basename(run)

        # Dummy Dataframe
        ER, F, LE, LR, seld_scr = None, None, None, None, None
        df = df.append({'group': group, 'seed': seed, 'run': tmp, 'checkpoint': model, 'pred_path': pred_output_format_files,
                        'ER': ER, 'F': F, 'LE': LE, 'LR': LR, 'seld_score': seld_scr}, ignore_index=True)

    return df

def aggregate_scores_bootstrap(df, row_id=None, num_classes=12, dcase2022=False, n_trials=10):
    from evaluation.dcase2022_metrics import cls_compute_seld_results, parameters
    params = parameters.get_params()

    # Compute just the DCASE 2021 final results
    if not dcase2022:
        score_obj = cls_compute_seld_results.ComputeSELDResults(params,
                                                                ref_files_folder="/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3/metadata_dev")
    else:
        score_obj = cls_compute_seld_results.ComputeSELDResults(params,
                                                                ref_files_folder="/m/triton/scratch/work/falconr1/sony/data_dcase2022/metadata_dev")
    print('')

    df_MACRO = pd.DataFrame(columns=['group', 'ER (median)', 'ER (ci_low)', 'ER (ci_hi)',
                                     'F (median)', 'F (ci_low)', 'F (ci_hi)',
                                     'LE (median)', 'LE (ci_low)', 'LE (ci_hi)',
                                     'LR (median)', 'LR (ci_low)', 'LR (ci_hi)',
                                     'seld_score (median)', 'seld_score (ci_low)', 'seld_score (ci_hi)', 'class_wise'])

    df_micro = pd.DataFrame(columns=['group', 'ER (median)', 'ER (ci_low)', 'ER (ci_hi)',
                                     'F (median)', 'F (ci_low)', 'F (ci_hi)',
                                     'LE (median)', 'LE (ci_low)', 'LE (ci_hi)',
                                     'LR (median)', 'LR (ci_low)', 'LR (ci_hi)',
                                     'seld_score (median)', 'seld_score (ci_low)', 'seld_score (ci_hi)'])

    groups = df['group'].unique()

    # Filter groups when running in parallel
    if row_id is not None:
        groups_list = groups[row_id:row_id+1]
    else:
        groups_list = groups

    for this_group in groups_list:
        paths = []
        print(f'Group = {this_group}')
        mask = df['group'] == this_group
        full_paths = df[mask]['pred_path'].tolist()
        print(f'Bootstrap analysis with {len(full_paths)} runs in group.')

        seld_metric_macro, seld_metric_micro = score_obj.get_SELD_Results_Bootstrap(full_paths, num_classes=num_classes, n_trials=n_trials)

        # ===================
        # Micro metrics
        ER_median, F_median, LE_median, LR_median, seld_scr_median = seld_metric_micro[0]
        ER_ci_low, F_ci_low, LE_ci_low, LR_ci_low, seld_scr_ci_low = seld_metric_micro[1]
        ER_ci_hi, F_ci_hi, LE_ci_hi, LR_ci_hi, seld_scr_ci_hi = seld_metric_micro[2]
        df_micro = df_micro.append({'group': this_group,
                                    'ER (median)': ER_median, 'ER (ci_low)': ER_ci_low, 'ER (ci_hi)': ER_ci_hi,
                                    'F (median)': F_median, 'F (ci_low)': F_ci_low, 'F (ci_hi)': F_ci_hi,
                                    'LE (median)': LE_median, 'LE (ci_low)': LE_ci_low, 'LE (ci_hi)': LE_ci_hi,
                                    'LR (median)': LR_median, 'LR (ci_low)': LR_ci_low, 'LR (ci_hi)': LR_ci_hi,
                                    'seld_score (median)': seld_scr_median, 'seld_score (ci_low)': seld_scr_ci_low,
                                    'seld_score (ci_hi)': seld_scr_ci_hi}, ignore_index=True)

        # ===================
        # MACRO metrics
        ER_median, F_median, LE_median, LR_median, seld_scr_median = seld_metric_macro[0]
        ER_ci_low, F_ci_low, LE_ci_low, LR_ci_low, seld_scr_ci_low = seld_metric_macro[1]
        ER_ci_hi, F_ci_hi, LE_ci_hi, LR_ci_hi, seld_scr_ci_hi = seld_metric_macro[2]
        df_MACRO = df_MACRO.append({'group': this_group,
                                    'ER (median)': ER_median, 'ER (ci_low)': ER_ci_low, 'ER (ci_hi)': ER_ci_hi,
                                    'F (median)': F_median, 'F (ci_low)': F_ci_low, 'F (ci_hi)': F_ci_hi,
                                    'LE (median)': LE_median, 'LE (ci_low)': LE_ci_low, 'LE (ci_hi)': LE_ci_hi,
                                    'LR (median)': LR_median, 'LR (ci_low)': LR_ci_low, 'LR (ci_hi)': LR_ci_hi,
                                    'seld_score (median)': seld_scr_median, 'seld_score (ci_low)': seld_scr_ci_low,
                                    'seld_score (ci_hi)': seld_scr_ci_hi, 'class_wise': None}, ignore_index=True)
    return df_MACRO, df_micro

def export_individual_csvs(df_MACRO, df_micro):
    if not os.path.exists(f'./csvs/{filename}'):
        os.mkdir(f'./csvs/{filename}')
    df_MACRO.to_csv(path_or_buf=f'./csvs/{filename}/df_{filename}_{params.chunk_id}_MACRO.csv')
    df_micro.to_csv(path_or_buf=f'./csvs/{filename}/df_{filename}_{params.chunk_id}_micro.csv')

def plot_metrics(df_MACRO, suptitle=''):
    stats = []
    for index, row in df_MACRO.iterrows():
        this_stat = {
            "label": row['group'],  # not required
            "mean": 0.5,  # not required
            "med": row['seld_score (median)'],
            "q1": row['seld_score (ci_low)'],
            "q3": row['seld_score (ci_hi)'],
            # "cilo": 5.3 # not required
            # "cihi": 5.7 # not required
            "whislo": row['seld_score (ci_low)'],  # required
            "whishi": row['seld_score (ci_low)'],  # required
            "fliers": []  # required if showfliers=True
        }
        stats.append(this_stat)
    stats_seld = []
    stats_er = []
    stats_f = []
    stats_le = []
    stats_lr = []
    for index, row in df_MACRO.iterrows():
        this_stat = {
            "label": row['group'],  # not required
            "mean": 0.5,  # not required
            "med": row['seld_score (median)'],
            "q1": row['seld_score (ci_low)'],
            "q3": row['seld_score (ci_hi)'],
            # "cilo": 5.3 # not required
            # "cihi": 5.7 # not required
            "whislo": row['seld_score (ci_low)'],  # required
            "whishi": row['seld_score (ci_low)'],  # required
            "fliers": []  # required if showfliers=True
        }
        stats_seld.append(this_stat)
        this_stat = {
            # "label": row['group'],  # not required
            "mean": 0.5,  # not required
            "med": row['ER (median)'],
            "q1": row['ER (ci_low)'],
            "q3": row['ER (ci_hi)'],
            # "cilo": 5.3 # not required
            # "cihi": 5.7 # not required
            "whislo": row['ER (ci_low)'],  # required
            "whishi": row['ER (ci_low)'],  # required
            "fliers": []  # required if showfliers=True
        }
        stats_er.append(this_stat)
        this_stat = {
            # "label": row['group'],  # not required
            "mean": 0.5,  # not required
            "med": row['F (median)'],
            "q1": row['F (ci_low)'],
            "q3": row['F (ci_hi)'],
            # "cilo": 5.3 # not required
            # "cihi": 5.7 # not required
            "whislo": row['F (ci_low)'],  # required
            "whishi": row['F (ci_low)'],  # required
            "fliers": []  # required if showfliers=True
        }
        stats_f.append(this_stat)
        this_stat = {
            # "label": row['group'],  # not required
            "mean": 0.5,  # not required
            "med": row['LE (median)'],
            "q1": row['LE (ci_low)'],
            "q3": row['LE (ci_hi)'],
            # "cilo": 5.3 # not required
            # "cihi": 5.7 # not required
            "whislo": row['LE (ci_low)'],  # required
            "whishi": row['LE (ci_low)'],  # required
            "fliers": []  # required if showfliers=True
        }
        stats_le.append(this_stat)
        this_stat = {
            # "label": row['group'],  # not required
            "mean": 0.5,  # not required
            "med": row['LR (median)'],
            "q1": row['LR (ci_low)'],
            "q3": row['LR (ci_hi)'],
            # "cilo": 5.3 # not required
            # "cihi": 5.7 # not required
            "whislo": row['LR (ci_low)'],  # required
            "whishi": row['LR (ci_low)'],  # required
            "fliers": []  # required if showfliers=True
        }
        stats_lr.append(this_stat)
    if True:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = False
        plt.close('all')
        fs = 10  # fontsize
        fig, axes_all = plt.subplots(nrows=1, ncols=5, figsize=(24, 4), sharey=True)
        # titles=[r'$\mathrm{ER}_{\mathrm{LD}}\downarrow$', r"$\mathrm{F}_{\mathrm{LD}}\uparrow$", r'$\mathrm{LE}_{\mathrm{CD}}\downarrow$', r'$\mathrm{LR}_{\mathrm{CD}}\uparrow$', r'$\rm{\mathcal{E}_{SELD}}\downarrow$' ]
        ctr = 0
        for axes, stats in zip(axes_all.flatten(), [stats_er, stats_f, stats_le, stats_lr, stats_seld]):
            coso = axes.bxp(stats, vert=False, patch_artist=True, widths=0.6, showcaps=False, showmeans=False, showfliers=True,
                            manage_ticks=True)
            axes.set_title(r"$\mathrm{ER}_{\mathrm{LD}}", fontsize=fs)
            # axes.set_title(titles[ctr], fontsize=fs)
            ctr += 1
            # fill with colors
            colors = ['pink', 'lightblue', 'lightgreen']
            cmap = mpl.cm.get_cmap('Dark2')
            ii = 0
            for box, median in zip(coso['boxes'], coso['medians']):
                box.set_facecolor(cmap(ii))
                median.set(linewidth=5, color=[0.0, 0.0, 0.0])
                ii += 1
            ii = 0
            for jj in range(0, len(coso['whiskers']), 2):
                print(jj)
                coso['whiskers'][jj].set(color=cmap(ii))
                coso['whiskers'][jj + 1].set(color=cmap(ii))
                ii += 1
            ii = 0
            # Move left and bottom spines outward by 10 points
            axes.spines.left.set_position(('outward', 10))
            axes.spines.bottom.set_position(('outward', 10))
            # Hide the right and top spines
            axes.spines.left.set_visible(False)
            axes.spines.right.set_visible(False)
            axes.spines.top.set_visible(False)
            # Only show ticks on the left and bottom spines
            axes.yaxis.set_ticks_position('none')
            axes.xaxis.set_ticks_position('bottom')
            axes.xaxis.set_tick_params(width=3)
            axes.spines.bottom.set_linewidth(2)
            axes.grid(color=[0.75, 0.75, 0.75], linestyle='-', linewidth=2, axis='x')
            plt.tight_layout(w_pad=4)
        plt.suptitle(suptitle)
        plt.show()
    if False:
        # testing my plot
        import seaborn as sns
        # Construct the DataFrame with MultiIndex:
        df = pd.DataFrame.from_dict({"Min Value": [2, 5, 3, 4, 1, 0],
                                     "Max Value": [10, 5, 8, 5, 11, 4]})
        index_arr = [["A", "A", "A", "B", "B", "B"],
                     ["Bad", "Good", "Good", "Bad", "Good", "Good"]]
        df.index = pd.MultiIndex.from_arrays(index_arr,
                                             names=["Category 1", "Category 2"])
        # Use groupby to find min and max for each category
        df_max = df.groupby(by=['Category 1', 'Category 2']).max()
        df_min = df.groupby(by=['Category 1', 'Category 2']).min()
        # Set the min values for the first column
        df_max.iloc[:, 0] = df_min.iloc[:, 0]
        # Plot the Boxplot (seems more appropriate then barplot)
        sns.boxplot(data=df_max.T)

if __name__=='__main__':
    # IMORTANT:
    # Manually copy the selected runs to the list_of_runs.txt
    # This could be for example, table_01_sXXXX or so
    #
    # So first, run download_models.sh   <----- manually update here the runs that I want
    # Then go to the tempeval directory
    # Copy the short name runs to -----> list_of_runs.txt     (this oculd be automated)
    # Then run this script
    #
    #
    # UPDATE This script now has 2 modes:
    # mode == 0: Call this to genereate the list_of_checkpoints.txt file, and then run the prediction with test data
    # PRediction is run with the seld_evaluation_tables.sh
    # mode == 1: Call this to print to console the CSV formatted results, raw format so the averaging will be done manually in excel or something
    # mode == 2: Call this to do the averaging directly here and print only the table
    #
    # Pass the path as arg

    params = get_parameters()

    print('Starting analysis of results for tables.')
    print(f'Mode : {params.mode}')
    print(f'dcase2022 : {params.dcase2022}')
    print(f'filename : {params.filename}')
    print(f'n_trials : {params.n_trials}')
    print(f'chunk_id : {params.chunk_id}')

    root = "/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3/model_monitor/tempeval"
    path = f'list_of_runs_{params.filename}.txt'

    filename = params.filename
    if True:
        pass

    else:
        path = "path_to_your_summaries"

        #root = '/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3/model_monitor/'
        #path = 'table01_s1234_baseline_w-rec:100.0_w-adv:0.0_ls_ls_G_lr:0.001_D_lr:0.1_nda_lam:1.0_nda_par:__thrshld_min:-1_thrshld_max:1000_curr-w-adv:0.0_curr-thrshld_min:0.0_curr-thrshld_max:0.0__20220402162257'

        root = "/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3/model_monitor/tempeval"
        root = "/m/triton/scratch/work/falconr1/sony/data_dcase2021_task3/model_monitor/tempeval"
        #path = 'table01_s1234_baseline-DAN-coord-threshold_2x_w-rec:100.0_w-adv:0.3_ls_ls_G_lr:0.001_D_lr:0.1_nda_l'


        path = 'list_of_runs_table03.txt'
        path = 'list_of_runs_dcase2022.txt'
        #path = 'list_of_runs_NDAALL.txt'
        #path = 'list_of_runs.txt'
        #path = 'list_of_runs_NDAnoise.txt'
        #path = 'list_of_runs_table03_dcase2022.txt'

        filename = 'table01_low_cDAN'
        filename = ''
        filename = '_table01_dcase2022_low_cDAN'
        filename = '_table03_dcase2022_low_cDAN'
        path = f'list_of_runs{filename}.txt'

    with open(os.path.join(root, path)) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    # Process each run
    all_outputs = []
    for line in lines:
        tmp = extract_top_checkpoints(root, os.path.join(root, line), top_k=1)
        if tmp is not None:
            if params.dcase2022:
                if 'dcase2022' not in tmp[0]:
                    continue
            else:
                if 'dcase2022' in tmp[0]:
                    continue
            all_outputs.extend(tmp)

    if params.mode == 0 or params.mode == '0':
        # Write to file
        print(' writing file')
        with open(os.path.join(root, f'list_of_checkpoints_{filename}.txt'), 'w') as f:
            for line in all_outputs:
                f.write(f'{line}\n')

    # Mean and stdev across seeds
    elif params.mode == 1 or params.mode == '1':
        df_MACRO, df_micro = run_evaluation(all_outputs, num_classes=13 if params.dcase2022 else 12, dcase2022=params.dcase2022)  # chang enum_calsses according to dataset
        df_MACRO_means, df_MACRO_stds = aggregate_scores(df_MACRO)
        df_micro_means, df_micro_stds = aggregate_scores(df_micro)

    # Median and CIs, using stratified bootstrat across seeds, and wavs
    elif params.mode == 2 or params.mode == '2':
        print('')
        print('')
        print(f'Running bootstrap with n_trials = {params.n_trials} ')
        num_classes = 13 if params.dcase2022 else 12  # chang enum_calsses according to dataset
        df_dummy = prepare_evaluation_bootstrap(all_outputs, num_classes=num_classes, dcase2022=params.dcase2022)
        df_MACRO, df_micro = aggregate_scores_bootstrap(df_dummy, num_classes=num_classes, dcase2022=params.dcase2022, n_trials=params.n_trials)
        #export_individual_csvs(df_MACRO, df_micro)

        suptitle = ''
        suptitle = 'MACRO, 2022, ntrials=10'
        plot_metrics(df_MACRO, suptitle=suptitle)


    elif params.mode == 3:
        if params.chunk_id > len(all_outputs):
            print(f'WARNING: chunk_id {params.chunk_id} is out of bounds. ')
            print('Exiting')
            import sys
            sys.exit(1)

        print('')
        print('')
        print(f'Running bootstrap with n_trials = {params.n_trials} ')
        num_classes = 13 if params.dcase2022 else 12  # chang enum_calsses according to dataset
        #df_dummy = prepare_evaluation_bootstrap(all_outputs[params.chunk_id:params.chunk_id+1], num_classes=num_classes, dcase2022=params.dcase2022)
        df_dummy = prepare_evaluation_bootstrap(all_outputs, num_classes=num_classes, dcase2022=params.dcase2022)
        df_MACRO, df_micro = aggregate_scores_bootstrap(df_dummy, row_id=params.chunk_id, num_classes=num_classes, dcase2022=params.dcase2022, n_trials=params.n_trials)
        export_individual_csvs(df_MACRO, df_micro)

    elif params.mode == 5:
        print(f'Collecting CSVS for filename {params.filename}')
        root = "/m/triton/scratch/work/falconr1/dcase2022/seld_dcase2022_ric"
        files = os.listdir(os.path.join(root, f'csvs/{filename}'))
        df_MACRO_list, df_micro_list = [], []
        for this_file in files:
            if not '.csv' in os.path.basename(this_file):
                continue
            if not f'df_{filename}' in os.path.basename(this_file):
                continue
            tmp = pd.read_csv(filepath_or_buffer=os.path.join(root, f'csvs/{filename}', this_file), index_col=0)
            if len(tmp) > 0:
                if 'MACRO' in os.path.basename(this_file):
                    df_MACRO_list.append(tmp)
                if 'micro' in os.path.basename(this_file):
                    df_micro_list.append(tmp)
        df_MACRO_out = pd.concat(df_MACRO_list, ignore_index=True)
        df_micro_out = pd.concat(df_micro_list, ignore_index=True)

        suptitle = ''
        #suptitle = 'micro, 2022, ntrials=1000'
        plot_metrics(df_micro_out, suptitle=suptitle)
        # Note, I should use debug mode here, and manually copy paste the DFs to my excel for formatting

    print("============================")
    print("Finished")

