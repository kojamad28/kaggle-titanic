from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import ClassificationModel
from pycaret.regression import RegressionModel
from pycaret.utils import check_metric

# Keys for the input_files and input_data dictionaries created by the DataImporter
KEY_TRAIN_DATA = 'train'
KEY_TEST_DATA = 'test'
KEY_SAMPLE_SUBMISSION_DATA = 'sample_submission'


class DataImporter:
    """
    Import csv files provided by Kaggle.

    Attributes
    ----------
    _input_dir : str, default './input/'
        Directory for the input files
    _train_file : str, default 'train.csv'
        CSV file of the dataset used for modeling
    _test_file : str, default 'test.csv'
        CSV file of the dataset used for predictions
    _sample_submission_file : 'sample_submission.csv'
        CSV file of an example format for submission

    Methods
    -------
    get_file_path
        Return the dictionary of the path of input files
    read
        Import the csv files as pandas.dataframes and return the dictionary of the dataframes.
    """
    
    def __init__(
            self, input_dir: str = './input/',
            train_file: str = 'train.csv', test_file: str = 'test.csv',
            sample_submission_file: str = 'sample_submission.csv', index_col: Union[str, int, None] = None
        ):
        self._train_file_path = input_dir + train_file
        self._test_file_path = input_dir + test_file
        self._sample_submission_file_path = input_dir + sample_submission_file
        self._index_col = index_col

    def get_file_path(self):
        """
        Return the dictionary of the input file path

        Returns
        -------
        input_files : dict
            Keys are strings of the data name (e.g, train, test) and values are strings of the file path.
            This object is used for autokeras data processing.
        """
        input_files = {
            KEY_TRAIN_DATA: self._train_file_path,
            KEY_TEST_DATA: self._test_file_path,
            KEY_SAMPLE_SUBMISSION_DATA: self._sample_submission_file_path
        }
        return input_files
        
    def read(self):
        """
        Import the csv files as pandas.dataframes and return the dictionary of the dataframes.

        Returns
        -------
        input_data : dict
            Keys are strings of the data name and values are pandas.DataFrame objects.
            This object is used for pycaret data processing.
        """
        df_train = pd.read_csv(self._train_file_path, index_col=self._index_col)
        print(f'Data for Modeling: {df_train.shape[0]} records, {df_train.shape[1]} features')
        df_test = pd.read_csv(self._test_file_path, index_col=self._index_col)
        print(f'Unseen Data for Predictions: {df_test.shape[0]} records, {df_test.shape[1]} features')
        df_sample_submission = pd.read_csv(self._sample_submission_file_path, index_col=self._index_col)
        input_data = {
            KEY_TRAIN_DATA: df_train,
            KEY_TEST_DATA: df_test,
            KEY_SAMPLE_SUBMISSION_DATA: df_sample_submission
        }
        return input_data


class PycaretEstimator:
    """
    Execute modeling and predictions using pycaret.
    
    Attributes
    ----------
    _mod
        Pycaret module used
    _train_data : pandas.DataFrame
        Dataset for modeling
    _test_data : pandas.DataFrame
        Dataset for predictions
    _sample_submit_data : pandas.DataFrame
        Example table for submission
    _created_target_col : str, default 'Label'
        Name of the new column for predicted values created by pycaret
    _created_score_col : str, default 'Score'
        Name of the new column for scores of the predicted classification created by pycaret
    _metrics_name_col
        Column name of the metric name of pycaret metrics table
    _metrics_display_name_col
        Column name of the metric display name of pycaret metrics table
    _target : str
        Target for predictions

    Methods
    -------
    _calculate_metrics
        Compute metrics from whole train dataset.
    _create_submit_data
        Create a dataframe for submission.
    check_baseline
        Execute predictions by simple parameters.
    """
    
    def __init__(self, mod, input_data: dict,
    created_target_col: str = 'Label', metrics_name_col: str = 'Name', metrics_display_name_col: str = 'Display Name',
    **kwargs):
        self._mod = mod
        self.train_data = input_data[KEY_TRAIN_DATA]
        self.test_data = input_data[KEY_TEST_DATA]
        self.sample_submit_data = input_data[KEY_SAMPLE_SUBMISSION_DATA]
        self.created_target_col = created_target_col
        self._metrics_name_col = metrics_name_col
        self._metrics_display_name_col = metrics_display_name_col
        self.target = list(set(self.train_data.columns) - set(self.test_data.columns))[0]
        self._mod.setup(data=self.train_data, target=self.target, **kwargs)

    @property
    def mod(self):
        return self._mod
        
    @property
    def metrics(self):
        return self._metrics
    
    def _calculate_metrics(self, model: Union[ClassificationModel, RegressionModel]):
        """
        Compute metrics from whole train dataset.

        Parameters
        ----------
        model : Union[ClassificationModel, RegressionModel]
            Estimator created by pycaret
        """
        self._metrics = {}
        df_metrics = self._mod.get_metrics()
        df_valid = self._mod.predict_model(model)
        y_true = df_valid.loc[:, self.target]
        y_pred = df_valid.loc[:, self.created_target_col]
        for metric in df_metrics.index:
            metric_name = df_metrics.loc[metric, self._metrics_name_col]
            metric_display_name = df_metrics.loc[metric, self._metrics_display_name_col]
            self._metrics[metric_display_name] = check_metric(y_true, y_pred, metric=metric_name)

    def _create_submit_data(self):
        """
        Create a dataframe for submission.

        Returns
        -------
        df_submit : pandas.DataFrame
            Data format for submission
        """
        df_submit = self.sample_submit_data.copy()
        df_submit.loc[:, self.target] = np.nan
        return df_submit
        
    def check_baseline(self, metric: str, tune_by: str = 'optuna', tuning_custom_grid: Optional[dict] = None):
        """
        Execute predictions by simple parameters.

        Parameters
        ----------
        metric : str
            Metric for selecting models and tuning
        tune_by : str, default 'optuna'
            Library used for tuning hyperparameters
        tuning_custom_grid : dict or None, default None
            Customized parameters for hypertuning

        Returns
        -------
        df_submit : pandas.DataFrame
            Dataframe for submission
        """
        best_model = self._mod.compare_models(sort=metric, verbose=False)
        tuned_best_model = self._mod.tune_model(
            best_model, optimize=metric, search_library=tune_by, custom_grid=tuning_custom_grid, verbose=False
        )
        final_best_model = self._mod.finalize_model(tuned_best_model)
        
        self._calculate_metrics(final_best_model)
        df_pred = self._mod.predict_model(final_best_model, data=self.test_data)
        df_submit = self._create_submit_data()
        df_submit.loc[:, self.target] = df_pred.loc[:, self.created_target_col]
        return df_submit

  
class DataLogger:
    """
    Log metrics and predicted values of models

    Attributes
    ----------
    _dataframe : pandas.DataFrame
        Summary table for the metrics values

    Methods
    -------
    add
        Insert data of a new model.
    _calculate_nrows_and_ncols
        Calculate numbers of rows and columns for subplots from given data.
    _get_ax
        Calculate the index of axis for a subplot.
    show_plots
        Display barplots for comparing data between models.
    """
    
    def __init__(self):
        self._dataframe = None
        
    @property
    def dataframe(self):
        return self._dataframe
        
    def add(self, metrics: dict, name: str):
        """
        Insert data of a new model.

        Parameters
        ----------
        metrics : dict
            Keys are data names, and values are data values.
        name : str
            Model name for comparison
        """
        if self._dataframe is None:
            self._dataframe = pd.DataFrame(columns=metrics.keys())
        self._dataframe.loc[name] = metrics
    
    def _calc_nrows_ncols(self, nplots: int, max_ncols: int):
        """
        Calculates numbers of rows and columns for subplots.

        Parameters
        ----------
        nplots : int
            Number of subplots in the figure
        max_ncols : int
            Maximum number of columns in the figure

        Returns
        -------
        nrows : int
            Calculated number of rows in the figure
        ncols : int
            Calculated number of columns in the figure
        """
        if isinstance(nplots, int) and isinstance(max_ncols, int):
            if nplots <= 0 or max_ncols <= 0:
                raise ValueError('nplots and max_ncols must be positive values.')
        else:
            raise TypeError('nplots and max_ncols must be positive integers.')

        if nplots < max_ncols:
            ncols = nplots
        else:
            ncols = max_ncols

        nrows = nplots // ncols + int(nplots % ncols != 0)

        return nrows, ncols

    def _get_ax(self, axes: matplotlib.axes.Axes, idx: int, nrows: int, ncols: int):
        """
        Gets axis object for the subplot.

        Parameters
        ----------
        axes : matplotlib.axes.Axes
            Axes objects to draw the plot onto
        idx : int
            Index number in plotted dataset
        nrows : int
            Number of rows in the figure
        ncols : int
            Number of columns in the figure

        Returns
        -------
        matplotlib.axes.Axes
        """

        if nrows > 1 and ncols > 1:
            return axes[idx // ncols, idx % ncols]
        elif nrows * ncols == 1:
            return axes
        else:
            return axes[idx]
        
    def show_plots(self, max_ncols: int= 3, base_size: int = 10):
        """
        Display barplots comparing data between models.

        Parameters
        ----------
        n_cols : int, default 3
            Number of columns on the figure
        base_size : float, default 10
            Size of a subplot
        """
        sns.set_context('poster', 1.5)
        nrows, ncols = self._calc_nrows_ncols(len(self._dataframe.columns), max_ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(base_size*nrows, base_size*ncols), tight_layout=True)
        for i, col in enumerate(self._dataframe.columns):
            ax_i = self._get_ax(axes, i, nrows, ncols)
            sns.barplot(data=self._dataframe.reset_index(), x='index', y=col, ax=ax_i)
            ax_i.set(title=col, xlabel='', ylabel='')
            ax_i.set_xticklabels(ax_i.get_xticklabels(), rotation=45, ha='right')
